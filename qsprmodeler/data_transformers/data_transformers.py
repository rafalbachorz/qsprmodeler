import sys
import json
from pathlib import Path
from tkinter.tix import X_REGION
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.Descriptors import qed
from rdkit.Chem import rdDistGeom
from rdkit.Chem import rdMolAlign
import pandas as pd
import numpy as np
from enum import Enum
from tqdm import tqdm

from .descriptors_definition import get_mordred_descriptors_definition, get_mordred_feature_names
import mordred
from mordred import descriptors, Calculator

ROOT_DIR = Path(__file__).parent.parent.parent
DEFAULT_BINARIZATION_TARGET = 1000

from auxiliary import loggers
l = loggers.get_logger(logger_name="logger")
import constants as co

class RemoveOutliers():
    def __init__(self, quantity, factor, target=True):
        l.info('Scaler constructor...')
        self.quantity = quantity
        self.factor = factor
        self.target = target
    
    @staticmethod
    def IQ_outlier_removal(data, quantity, factor):
        q1 = data[quantity].quantile(0.25)
        q3 = data[quantity].quantile(0.75)
        IQR = q3 - q1
        lb = q1 - factor * IQR
        up = q3 + factor * IQR
        l.info("IQR distance: "+str(IQR))
        return data[(data[quantity] > lb) & (data[quantity] < up)]

    def fit(self, X, y):
        l.info('RemoveOutliers fit...')
        if self.target:
            from_what = y
        else:
            from_what = X 
        self.__survived = self.IQ_outlier_removal(from_what, quantity=self.quantity, factor=self.factor).index
        return self
    
    def transform(self, X, y=None):
        l.info('RemoveOutliers transform...')
        data_wo = X.loc[self.__survived, :]
        return data_wo

    @property
    def survived(self):
        return self.__survived
    
class RemoveTargetOutliers():
    def __init__(self, factor):
        l.info('RemoveTargetOutliers constructor...')
        self.factor = factor
        self.survived = None
        
    @staticmethod
    def IQ_outlier_removal(data, quantity, factor):
        q1 = data[quantity].quantile(0.25)
        q3 = data[quantity].quantile(0.75)
        IQR = q3 - q1
        lb = q1 - factor * IQR
        up = q3 + factor * IQR
        l.info("IQR distance: "+str(IQR))
        return data[(data[quantity] > lb) & (data[quantity] < up)]

    def fit(self, y):
        l.info('RemoveTargetOutliers fit...')
        q1 = y.quantile(0.25)
        q3 = y.quantile(0.75)
        IQR = q3 - q1
        self.lb = q1 - self.factor * IQR
        self.up = q3 + self.factor * IQR        
        self.survived = y[(y > self.lb) & (y < self.up)].index
        return self
    
    def transform(self, y):
        l.info('RemoveTargetOutliers transform...')
        y_wo = y.loc[self.survived]
        return y_wo

class TurnSmilesIntoFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, calculate_fps=True, calculate_descriptors=True, 
                 fp_type="morgan", fingerprint_size=1024, chosen_descriptors=None, meta_descriptors=None, opt_3d=False) -> None:
        self.calculate_fps = calculate_fps
        self.calculate_descriptors = calculate_descriptors
        self.fp_type = fp_type
        self.fingerprint_size = fingerprint_size
        self.chosen_descriptors = chosen_descriptors
        self.meta_descriptors = meta_descriptors
        self.opt_3d = opt_3d
        self.fp_column_names = ['fp_'+str(idx) for idx in range(self.fingerprint_size)]      
        super().__init__()

    def __calculate_fps(self, mol_objs, index=None):
        if self.fp_type == "topological":
            fps = [Chem.RDKFingerprint(x, fpSize=self.fingerprint_size) for x in mol_objs]
        elif self.fp_type == "morgan":
            fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=self.fingerprint_size) if x is not None else None for x in mol_objs]
        elif self.fp_type == "maccs":
            fps = [MACCSkeys.GenMACCSKeys(x) for x in mol_objs]
            
        fps_bin = []
        for item in fps: 
            if item is not None: 
                fps_bin.append(DataStructs.BitVectToText(item))
            else:
                fps_bin.append([None for idx in range(self.fingerprint_size)])

        tmp = []
        for idx in range(len(fps_bin)): tmp.append([item for item in fps_bin[idx]])

        tmp = [[int(item2) if item2 is not None else None for item2 in item1] for item1 in fps_bin]
        tmp = pd.DataFrame(tmp)

        tmp.columns = self.fp_column_names
        if index is not None: tmp = tmp.set_index(index)
        return tmp

    def __confgen(self, mol, numConfs=1, maxAttempts=1000, pruneRmsThresh=0.1, useExpTorsionAnglePrefs=True, 
                  useBasicKnowledge=True, enforceChirality=True, randomSeed=42):
        mol = Chem.AddHs(mol, addCoords=True)
        param = rdDistGeom.ETKDGv2()
        cids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=numConfs, maxAttempts=maxAttempts, pruneRmsThresh=pruneRmsThresh, 
                                            useExpTorsionAnglePrefs=useExpTorsionAnglePrefs, useBasicKnowledge=useBasicKnowledge,
                                            enforceChirality=enforceChirality, randomSeed=randomSeed)
        
        mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=512, numThreads=8, mmffVariant='MMFF94s')

        res = [] 
        for cid in cids:
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
            e = ff.CalcEnergy()
            res.append((cid, e))
        rdMolAlign.AlignMolConformers(mol)
        
        energies = [e for i, e in res]
        
        confs = mol.GetConformers()
        for c, e in zip(confs, energies):
            c.SetProp('Energy', str(e))
        return mol

    def __is_morder_missing(self, x):
        return np.nan if type(x) == mordred.error.Missing or type(x) == mordred.error.Error else x

    def __calculate_descriptors(self, mol_objs, index=None):
        
        if self.chosen_descriptors is None:
            to_be_calculated = descriptors
        else:
            to_be_calculated = self.chosen_descriptors

        calc = Calculator(to_be_calculated, ignore_3D=True)
        molecular_descriptors = calc.pandas(mols=mol_objs, nproc=1)

        # simple processing, rather too offensife
        molecular_descriptors = molecular_descriptors.applymap(self.__is_morder_missing)
        molecular_descriptors = molecular_descriptors.dropna(axis=1, how='any')
        
        self.__descriptor_columns = molecular_descriptors.columns

        if index is not None:
            molecular_descriptors.index = index

        for column in molecular_descriptors.columns:
            if molecular_descriptors[column].dtype == bool: molecular_descriptors[column] = molecular_descriptors[column].astype(int)

        return molecular_descriptors

    def __calculate_meta_descriptors(self, mol_objs, index=None):
        if "QED" in self.meta_descriptors:
            qeds = [qed(mol) for mol in mol_objs]
            return pd.DataFrame(qeds, columns=["QED"], index = index)

    def __check_mols_consistency(self, mol_objs):
        none_idx = []
        for idx, item in enumerate(mol_objs):
            if item is None: 
                none_idx.append(idx)
                mol_objs.pop(idx)
        return mol_objs, none_idx

    def fit(self, X, y=None):
        l.info('CreateFPS fit...')
        return self
    
    def transform(self, X: pd.Series, y=None):
        l.info('CreateFPS transform...')
            
        mol_objs = [Chem.MolFromSmiles(item) for item in X.tolist()]
        mol_objs, none_idx = self.__check_mols_consistency(mol_objs)
        
        if self.opt_3d:
            #mol_objs_ = []
            #iii = 0
            #for mol in mol_objs:
            #    print(iii)
            #    mol_objs_.append(self.__confgen(mol))
            #    iii  += 1
            mol_objs = [self.__confgen(mol) for mol in tqdm(mol_objs)]
        if none_idx: X = X.drop(none_idx)

        if len(mol_objs) > 0:
            if self.calculate_fps:
                fps = self.__calculate_fps(mol_objs=mol_objs, index=X.index)
            else:
                fps = None
            if self.calculate_descriptors:
                if self.chosen_descriptors is not None:
                    molecular_descriptors = self.__calculate_descriptors(mol_objs=mol_objs, index=X.index)
                else:
                    molecular_descriptors = None
                if self.meta_descriptors is not None:
                    meta_descriptor = self.__calculate_meta_descriptors(mol_objs=mol_objs, index=X.index)
                else:
                    meta_descriptor = None
                descriptors_all = pd.concat([molecular_descriptors, meta_descriptor], axis=1)
            else:
                descriptors_all = None

            molecular_features = pd.concat([descriptors_all, fps], axis=1)
        return molecular_features
    
    @property
    def FP_column_names(self):
        return self.fp_column_names
    
class CreateFPS(BaseEstimator, TransformerMixin):
    def __init__(self, smiles_codes: pd.Series, fp_type="morgan", fingerprint_size=1024, first_step=False):
        l.info('CreateFPS constructor...')
        self.smiles_codes = smiles_codes
        self.fp_type = fp_type
        self.fingerprint_size = fingerprint_size
        self.first_step = first_step
        self.fp_column_names = ['fp_'+str(idx) for idx in range(self.fingerprint_size)]
    
    def __calculate_fps(self, mol_objs, index=None):
        if self.fp_type == 'topological':
            fps = [Chem.RDKFingerprint(x, fpSize=self.fingerprint_size) for x in mol_objs]
        elif self.fp_type == 'morgan':
            fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=self.fingerprint_size) if x is not None else None for x in mol_objs]
            
        fps_bin = []
        for item in fps: 
            if item is not None: 
                fps_bin.append(DataStructs.BitVectToText(item))
            else:
                fps_bin.append([None for idx in range(self.fingerprint_size)])

        tmp = []
        for idx in range(len(fps_bin)): tmp.append([item for item in fps_bin[idx]])

        tmp = [[int(item2) if item2 is not None else None for item2 in item1] for item1 in fps_bin]
        tmp = pd.DataFrame(tmp)

        tmp.columns = self.fp_column_names
        if index is not None: tmp = tmp.set_index(index)
        return tmp

    def fit(self, X, y=None):
        l.info('CreateFPS fit...')
        return self

    def __check_mols_consistency(self):
        none_idx = []
        for idx, item in enumerate(self.mol_objs):
            if item is None: 
                none_idx.append(idx)
                self.mol_objs.pop(idx)
        return none_idx

    def transform(self, X: pd.Series, y=None):
        l.info('CreateFPS transform...')
        if self.first_step: self.smiles_codes = X.copy()
            
        self.mol_objs = [Chem.MolFromSmiles(item) for item in self.smiles_codes.tolist()]
        none_idx = self.__check_mols_consistency()
        if none_idx: self.smiles_codes = self.smiles_codes.drop(none_idx)

        if len(self.mol_objs) > 0:
            fps = self.__calculate_fps(mol_objs=self.mol_objs, index=self.smiles_codes.index)
            if self.first_step:
                entire_df = fps
            else:
                entire_df = pd.concat([X, fps], axis=1)   
        else:
            entire_df = None
        return entire_df

    @property
    def Smiles(self):
        return self.smiles_codes
    
    @Smiles.setter
    def Smiles(self, value):
        self.smiles_codes = value

class CreateDescriptors(BaseEstimator, TransformerMixin):
    def __init__(self, smiles_codes: pd.Series, chosen_descriptors=None, meta_descriptors=None, first_step=False):
        l.info('CreateDescriptors constructor...')
        self.smiles_codes = smiles_codes
        # at the moment not used, but prepared for future selective descriptor calculation
        self.chosen_descriptors = chosen_descriptors
        self.meta_descriptors = meta_descriptors
        self.first_step = first_step

    def __is_morder_missing(self, x):
        return np.nan if type(x) == mordred.error.Missing or type(x) == mordred.error.Error else x

    def __calculate_descriptors(self, mol_objs, index=None):
        
        if self.chosen_descriptors is None:
            to_be_calculated = descriptors
        else:
            to_be_calculated = self.chosen_descriptors

        calc = Calculator(to_be_calculated, ignore_3D=True)
        molecular_descriptors = calc.pandas(mol_objs)

        # simple processing, rather too offensife
        molecular_descriptors = molecular_descriptors.applymap(self.__is_morder_missing)
        molecular_descriptors = molecular_descriptors.dropna(axis=1, how='any')
        
        self.__descriptor_columns = molecular_descriptors.columns

        if index is not None:
            molecular_descriptors.index = index

        for column in molecular_descriptors.columns:
            if molecular_descriptors[column].dtype == bool: molecular_descriptors[column] = molecular_descriptors[column].astype(int)

        return molecular_descriptors

    def __calculate_meta_descriptors(self, mol_objs, index=None):
        if "QED" in self.meta_descriptors:
            qeds = [qed(mol) for mol in mol_objs]
            return pd.DataFrame(qeds, columns=["QED"], index = index)

    def fit(self, X, y=None):
        l.info('CreateDescriptors fit...') 
        return self

    def __check_mols_consistency(self):
        none_idx = []
        for idx, item in enumerate(self.mol_objs):
            if item is None: 
                none_idx.append(idx)
                self.mol_objs.pop(idx)
        return none_idx

    def transform(self, X, y=None):
        l.info('CreateDescriptors transform...')
        if self.first_step: self.smiles_codes = X.copy()
        
        self.mol_objs = [Chem.MolFromSmiles(item) for item in self.smiles_codes.tolist()]
        none_idx = self.__check_mols_consistency()
        if none_idx: self.smiles_codes = self.smiles_codes.drop(none_idx)

        #entire_df = X
        if len(self.mol_objs) > 0:
            if self.chosen_descriptors is not None:
                molecular_descriptors = self.__calculate_descriptors(mol_objs=self.mol_objs, index=self.smiles_codes.index)
            else:
                molecular_descriptors = None
            if self.meta_descriptors is not None:
                meta_descriptor = self.__calculate_meta_descriptors(mol_objs=self.mol_objs, index=self.smiles_codes.index)
            else:
                meta_descriptor = None
                
            descriptors_all = pd.concat([molecular_descriptors, meta_descriptor], axis=1)
            if self.first_step:
                entire_df = descriptors_all
            else:
                entire_df = pd.concat([X, descriptors_all], axis=1)
        else:
            entire_df = None
        return entire_df

    @property
    def descriptor_columns(self):
        return self.__descriptor_columns

class DataScaling(BaseEstimator, TransformerMixin):
    def __init__(self, scaler, scaling_features):
        l.info('Scaler constructor...')
        self.scaler = scaler
        self.scaling_features = scaling_features
    
    def fit(self, X, y=None):
        l.info('Scaler fit...')
        self.remaining_features = list(set(X.columns.to_list()) - set(self.scaling_features))
        self.scaler.fit(X[self.scaling_features])
        return self
    
    def transform(self, X, y=None):
        l.info('Scaler transform...')

        scaled_data = self.scaler.transform(X[self.scaling_features])
        scaled_data = pd.DataFrame(scaled_data, columns=self.scaling_features, index=X.index)
        entire_data = pd.concat([scaled_data, X[self.remaining_features]], axis=1)
        all_features = sorted(entire_data.columns.tolist())
        return entire_data[all_features]
    
    @property
    def get_scaler(self):
        return self.scaler
    
class DataPCA(BaseEstimator, TransformerMixin):
    def __init__(self, pca_features_names: list, n_components=None):
        l.info('PCA constructor...')
        self.pca_features_names = pca_features_names
        if n_components:
            self.n_components = n_components
        else:
            self.n_components = len(pca_features_names)
            
        self.pca_feature_names = ["pca_"+str(iii) for iii in range(self.n_components)]
    
    def fit(self, X: pd.DataFrame, y=None):
        l.info('Scaler fit...')
        if self.n_components > X.shape[0]:
            self.n_components = X.shape[0]
            self.pca_feature_names = ["pca_"+str(iii) for iii in range(self.n_components)]
            
        self.pca = PCA(n_components=self.n_components, random_state=co.RANDOM_STATE)
        self.pca.fit(X[self.pca_features_names])
        return self
    
    def transform(self, X, y=None):
        l.info('PCA transform...')
        pca_features = pd.DataFrame(self.pca.transform(X[self.pca_features_names]), columns=self.pca_feature_names, index=X.index)
        new_features = X.drop(self.pca_features_names, axis=1)
        new_features = pd.concat([new_features, pca_features], axis=1)
        return new_features
    
    @property
    def PCA_feature_names(self):
        return self.pca_feature_names
    
    @property
    def PCA_model(self) -> PCA:
        return self.pca
    
class DataSupplement(BaseEstimator, TransformerMixin):
    def __init__(self, supplement_file: str, index_column: str):
        self.supplement_file = supplement_file
        self.index_column = index_column

    def fit(self, X:pd.DataFrame, y=None):
        self.suplement_data = pd.read_csv(self.supplement_file, index_col=self.index_column)

    def transform(self, X:pd.DataFrame, y=None):
        all_data = pd.concat([X, self.suplement_data], axis=1, join="inner")
        return all_data

def infer_PCA(columns: list) -> list:
    pca_features = []
    for feature in columns:
        if feature.startswith("pca_"): pca_features.append(feature)
    return pca_features

class DataRemoveHighlyCorrelated(BaseEstimator, TransformerMixin):
    def __init__(self, corr_thresnold, considered_features, infer_PCA):
        self.corr_thresnold = corr_thresnold
        self.considered_features = considered_features
        self.infer_PCA = infer_PCA
        self.to_drop = None
        self.corr_matrix = None
        self.susrvived_features = None
        
    def fit(self, X: pd.DataFrame, y=None):
        l.info('HighlyCorrelated fit...')
        # check if PCA is present in the pipeline
        return self
        
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        l.info('HighlyCorrelated transform...')
        if self.infer_PCA: self.considered_features += infer_PCA(X.columns.to_list())
        self.corr_matrix = X[self.considered_features].corr().abs()
        upper = self.corr_matrix.where(np.triu(np.ones(self.corr_matrix.shape), k=1).astype(bool))
        self.to_drop = [column for column in upper.columns if any(upper[column] > self.corr_thresnold)]
        survivals = X.drop(self.to_drop, axis=1)
        self.susrvived_features = survivals.columns.to_list()
        return survivals
    
    
    @property
    def correlated(self):
        return self.to_drop
    
    @correlated.setter
    def correlated(self, value):
        self.to_drop = value
        
    @property
    def Corrmatrix(self):
        return self.corr_matrix

    @property
    def SurvivedFeatures(self):
        return self.susrvived_features
    
    @SurvivedFeatures.setter
    def SurvivedFeatures(self, value):
        self.susrvived_features = value


class TransformTarget():
    #def __init__(self, divisor=1000, additive=1):
    def __init__(self, divisor=1000000000, additive=0, multiplicative=-1.0):
        self.__divisor = divisor
        self.__additive = additive
        self.__multiplicative = multiplicative
    
    def transform(self, y: pd.Series) -> pd.Series:
        y_tra = self.__multiplicative * np.log10(y/self.__divisor + self.__additive)
        #y_tra = np.log(y/self.__divisor + self.__additive)
        return pd.Series(y_tra)

    def inverse_transform(self, y_tra):
        y = (np.exp(y_tra) - self.__additive) * self.__divisor
        return pd.Series(y)

    @property
    def divisor(self):
        return self.__divisor
    
    @property
    def additive(self):
        return self.__additive
    
class BinarizeTarget():
    def __init__(self, threshold=DEFAULT_BINARIZATION_TARGET):
        self._threshold = threshold
    
    def fit(self, y):
        return self
    
    def transform(self, y: pd.Series) -> pd.Series:
        y_tra = y < self._threshold
        return pd.Series(y_tra)

    def inverse_transform(self, y_tra):
        raise NotImplemented

    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self, value):
        self._threshold = value

def prepare_molecular_descriptors_classes(descriptor_types: list, portfolio: dict) -> list:
    descriptor_classes = []
    for item in descriptor_types:
        try: 
            descriptor_classes += portfolio[item]
        except KeyError:
            l.error("Missing descriptor group: "+str(item))
    return descriptor_classes

def check_PCA_in_pipeline(steps):
    PCA_features = []
    for key, instance in steps:
        if key == "PCA": PCA_features = instance.PCA_feature_names
    return PCA_features

def check_if_step_present_in_PCA(steps, step_label):
    present = False
    for label, _ in steps:
        if label == step_label:
            present = True
            break
    return present

def create_pipeline(pipeline_configuration: dict):
    steps = []
    fp_features = None 
    descriptors_definition, molecular_meta_descriptors_definition = get_mordred_descriptors_definition()
    molecular_descriptors_feature_names, molecular_meta_descriptors_feature_name = get_mordred_feature_names()
    md_features = []

    fp_present = False
    md_present = False
    for step, configuration in pipeline_configuration.items():
        if configuration["proceed"] == "no": continue
        if step == "Calculate_molecular_features":
            descriptor_types = configuration["descriptors_types"]
            meta_descriptor_types = configuration["meta_descriptors_types"]
            md_features += prepare_molecular_descriptors_classes(descriptor_types=descriptor_types, portfolio=molecular_descriptors_feature_names)
            md_features += prepare_molecular_descriptors_classes(descriptor_types=meta_descriptor_types, portfolio=molecular_meta_descriptors_feature_name)
            # TODO: utilize the information about the considered features, they could/should(?) be passed to the scaping/pca classes
            descriptor_classes = prepare_molecular_descriptors_classes(descriptor_types=descriptor_types, portfolio=descriptors_definition)
            meta_descriptor_classes = prepare_molecular_descriptors_classes(descriptor_types=meta_descriptor_types, portfolio=molecular_meta_descriptors_definition)
            
            fp_present = configuration["calculate_fps"]=="yes"
            md_present = configuration["calculate_descriptors"]=="yes"
            # optional parameters
            if "opt_3d" in configuration.keys():
                opt_3d = configuration["opt_3d"]=="yes"
            else:
                opt_3d = None
                
            MF_creator = TurnSmilesIntoFeatures(calculate_fps=fp_present, calculate_descriptors=md_present,
                                                fp_type=configuration["fp_type"], fingerprint_size=configuration["fingerprint_size"], 
                                                chosen_descriptors=descriptor_classes, meta_descriptors=meta_descriptor_classes, opt_3d=opt_3d)
            steps.append((configuration["label"], MF_creator))
            fp_features = MF_creator.FP_column_names
        elif step == "Scaling":
            scaler = StandardScaler()
            scaling_features = configuration["scaling_features"]
            Scaling = DataScaling(scaler, scaling_features)
            steps.append((configuration["label"], Scaling))
        elif step == "PCA":
            # potentially the md or/and fp are replaced with their PCA eqauivalent
            if configuration["proceed"] == "no": continue
            if "n_components" in configuration.keys():
                n_components = configuration["n_components"]
            else:
                n_components = None
            to_be_considered = []
            if configuration["apply_to_fp"] == "yes":
                to_be_considered += fp_features
                fp_present = False
            if configuration["apply_to_md"] == "yes":
                to_be_considered += md_features
                md_present = False
            PCA_transformer = DataPCA(to_be_considered, n_components=n_components)
            steps.append((configuration["label"], PCA_transformer))
        elif step == "Supplement":
            supplement_file = configuration["supplement_file"]
            index_column = configuration["index_column"]
            Supplementer = DataSupplement(supplement_file=supplement_file, index_column=index_column)
            steps.append((configuration["label"], Supplementer))
        elif step == "Corr":
            threshold = configuration["corr_threshold"]
            to_be_inspected = []
            infer_PCA = False
            if configuration["apply_to_md"] == "yes" and md_present:
                to_be_inspected += md_features
            if configuration["apply_to_fp"] == "yes" and fp_present:
                to_be_inspected += fp_features
            if configuration["apply_to_pca"] == "yes":
                infer_PCA = True
            Corr = DataRemoveHighlyCorrelated(threshold, to_be_inspected, infer_PCA)
            steps.append((configuration["label"], Corr))
        else:
            raise NotImplementedError

    pipeline = Pipeline(steps=steps)
    return pipeline

def create_molecular_features(pipeline, smiles_codes: pd.Series):
    X = pipeline[0].fit_transform(smiles_codes)
    return X

def process_taget(pipeline_configuration, y):
    for step, configuration in pipeline_configuration.items():
        if configuration["proceed"] == "no": continue
        if step == "Target_transform":
            Target_transformer = TransformTarget()
            y = Target_transformer.transform(y)
        elif step == "Target_binarization":
            Binarizer = BinarizeTarget(threshold=configuration["threshold"])
            y = Binarizer.transform(y)
        elif step == "O_removal":
            O_removal = RemoveTargetOutliers(factor=configuration["factor"])
            O_removal.fit(y)
            y = O_removal.transform(y)
        else:
            raise NotImplementedError
    return y    

def process_molecular_features(pipeline, X):
    if len(pipeline) > 1: X = pipeline[1:].transform(X)
    # some rows can drop during the fit/transform of the data
    return X

def validate_smiles(smiles_codes: pd.Series) -> list:
    none_idx = []
    mol_objs = [Chem.MolFromSmiles(mol) for mol in smiles_codes.to_list()]
    for idx, item in zip(smiles_codes.index.to_list(), mol_objs):
        if item is None: none_idx.append(idx)
    return none_idx

def read_in_pipeline(pipeline_file, pipeline_directory=None):
    if pipeline_directory is not None:
        pipeline_file = (pipeline_directory/pipeline_file).absolute().as_posix()
    with open(pipeline_file, "r") as f:
        pipeline = json.load(f)
    return pipeline

# class ColumnNames(Enum):
#     SMILES_COLUMN = "canonical_smiles_get_levels"
#     TARGET_COLUMN = "experimental_value"

def data_load(configuration: dict) -> pd.DataFrame:
    """_summary_

    Args:
        configuration (dict): configuration of the data preprocessing

    Raises:
        NotImplementedError: 

    Returns:
        pd.DataFrame: the data for the QSPR/QSAR
    """
    file_name = Path(configuration["data_file"])
    train_data = pd.read_csv((file_name).absolute().as_posix())

    molecule_column = configuration["molecule_column"]
    experimental_data_prefix = configuration["experimental_data_prefix"]

    activity_columns = [experimental_data_prefix+"_"+str(iii) for iii in range(configuration["max_level_activity"])]

    try:
        train_data["target_std"] = train_data[activity_columns].apply(lambda x: np.std(x), axis=1)
    except:
        raise ValueError
    
    # here we exclude the molecules for which the std of the available measurements is significant 
    target_std_threshold = configuration["std_threshold"]
    try:
        train_data = train_data[train_data["target_std"] < target_std_threshold]
    except:
        pass  
    
    target_column = configuration["target_column"]
    try:
        strategy = configuration["strategy"]
        if strategy == "min":
            train_data[target_column] = train_data[activity_columns].apply(lambda x: np.min(x), axis=1)
        elif strategy == "max":
            train_data[target_column] = train_data[activity_columns].apply(lambda x: np.max(x), axis=1)
        elif strategy == "median":
            train_data[target_column] = train_data[activity_columns].apply(lambda x: np.median(x[~np.isnan(x)]), axis=1)
        elif strategy == "mean":
            train_data[target_column] = train_data[activity_columns].apply(lambda x: np.mean(x), axis=1)
        else:
            raise NotImplementedError
    except:
        pass
    
    return train_data.loc[:, [molecule_column, target_column]]