import copy
import pickle
from tabnanny import verbose
import dill as pickle

from prometheus_client import Enum
import data_transformers as dt
#from data_transformers import TurnSmilesIntoFeatures, DataScaling, TransformTarget, BinarizeTarget, RemoveTargetOutliers, RemoveOutliers
from mordred import Calculator, descriptors, RingCount, SLogP, RingCount, HydrogenBond, RotatableBond, Weight, TopoPSA, Lipinski
from auxiliary import loggers

import os
import tempfile
import abc
from typing import Any, NoReturn
from functools import partial

import pandas as pd
import numpy as np
import random as rn

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, InputLayer, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras import initializers
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
import tensorflow as tf
import tensorflow.keras.metrics as metrics

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, roc_auc_score, recall_score, \
precision_score, accuracy_score, precision_recall_curve, mean_squared_error


from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.linear_model import BayesianRidge, LinearRegression, LogisticRegression, \
RidgeClassifier, RidgeClassifierCV, Lasso, Ridge
from sklearn.svm import SVC, SVR
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.pipeline import Pipeline

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent.parent
import constants as co

l = loggers.get_logger(logger_name="logger")

class Prediction_Type(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    
class Prediction_Methodology(Enum):
    XGBOOST = "M_XGBoost"
    MLP = "M_MLP"
    RF = "M_RandomForest"
    BAGGING = "M_Bagging"
    SVM = "M_SVM"
    RIDGE = "M_Ridge"
    
class Model_Status(Enum):
    UNKNOWN = "unknown"
    PRODUCTION = "production"
    

class Model(metaclass=abc.ABCMeta):
    def __init__(self, prediction_type: Prediction_Type, hyperparameters: dict, hyperparameter_types: dict, space: dict, pipeline_configuration: dict):
        """
        The Abstract class defining the interface

        Args:
            prediction_type (_type_): the class of predictive service, regression or classification
            hyperparameters (_type_): the set of hyperparameters
            hyperparameter_types (_type_): the set of hyperparameter types
            space (_type_): the space of hyperparameters
            pipeline_configuration (_type_): the parametrization of the data creation/preprocessing
        """        
        self._prediction_type = prediction_type
        self._hyperparameters = hyperparameters.copy()
        self._default_hyperparameters = hyperparameters.copy()
        self._hyperparameter_types = hyperparameter_types
        self._space = space
        self._pipeline_configuration = pipeline_configuration
        
        self._validate = False
        self._model = None
        self._model_status = Model_Status.UNKNOWN
        self._debug = False
        self._random_seed = 42
        self._data = None
        self._molecular_features = None
        self._optimization_history = None
        self._measaures = None
        self._default_measures = None
        self._feature_set = None
        self._GPU_support = False
        self._training_params = None
        
        if pipeline_configuration:
            self._pipeline = dt.create_pipeline(pipeline_configuration["feature_transform"])
        else:
            self._pipeline = None
       
    def Initialize_Seeds(self, random_seed: int) -> NoReturn:
        """
        Random seeds initialization

        Args:
            random_seed (int): random seed
        """        
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        rn.seed(random_seed)
        os.environ["HYPEROPT_FMIN_SEED"] = str(random_seed)

    def Test_Model(self, y_test: pd.Series, y_pred: pd.Series, threshold: float) -> dict:
        """Calculate the quality measures of the model

        Args:
            y_test (pd.Series): the ground truth
            y_pred (pd.Series): the prediction
            threshold (float): the threshold for target binarization (relevant for the classifiers)

        Returns:
            dict: the dictionary with measures
        """        
        measures = {}
        if self._prediction_type == Prediction_Type.CLASSIFICATION:
            y_pred_crisp = y_pred > threshold
            measures["precision"] = precision_score(y_test, y_pred_crisp)
            measures["recall"] = recall_score(y_test, y_pred_crisp)
            measures["precision_recall_ave"] = (measures["precision"] + measures["recall"])/2.0
            measures["average_precision"] = average_precision_score(y_test, y_pred)
            measures["roc_auc"] = roc_auc_score(y_test, y_pred)
            measures["pr_re_ap_rocauc_ave"] = (measures["precision"]+measures["recall"]+measures["average_precision"]+measures["roc_auc"])/4.0
            measures["f1"] = f1_score(y_test, y_pred_crisp)
        elif self._prediction_type == Prediction_Type.REGRESSION:
            try:
                measures["mse"] = mean_squared_error(y_test, y_pred)
            except ValueError:
                measures["mse"] = co.ERROR_GOAL_FUCTION
        return measures

    def _refresh_hyperparameters(self, hps_addons: dict) -> NoReturn:
        """
        Refreshing the hyperparameters, to be fixed

        Args:
            hps_addons (dict): the current hyperparameters

        Returns:
            NoReturn:
        """        
        # update the default hyperparameters
        if hps_addons:
            for key, value in hps_addons.items():
                    if key in self._hyperparameters: self._hyperparameters[key] = value
        
        # get the right types of the hyperparameters
        for key, item in self._hyperparameters.items(): self._hyperparameters[key] = self._hyperparameter_types[key](self._hyperparameters[key])
    
    def Train_CV(self, hyperparameters: dict, n_outer: int, n_cv: int, default_hyperparameters=False) -> pd.Series:
        """
        The n-fold cross-validation carried multiple times
        It is assumed that the training set is provided into the class (self._data)

        Args:
            hyperparameters (dict): current hyperparameters
            n_outer (int): how many times the CV should be done
            n_cv (int): how many folds within the CV procedure

        Returns:
            pd.Series: The cross-validated predictions
        """
        self._refresh_hyperparameters(hyperparameters)
        y_pred_final = np.zeros(self.Data["targets_train"].shape)
        
        for i_outer in range(n_outer):        
            kf = KFold(n_splits=n_cv, random_state=self._random_seed+i_outer, shuffle=True)
            y_pred_fold = np.zeros(self.Data["targets_train"].shape)
            for n, (train_idx, test_idx) in enumerate(kf.split(self._molecular_features)):
                
                X_train = self._molecular_features.iloc[train_idx, :]
                X_test = self._molecular_features.iloc[test_idx, :]
                y_train = self.Data["targets_train"].iloc[train_idx]
                y_test = self.Data["targets_train"].iloc[test_idx]
                
                # do not recreate molecular features
                if len(self.Pipeline) > 1: self.Pipeline[1:].fit(X=X_train, y=y_train)
                
                X_train = dt.process_molecular_features(pipeline=self.Pipeline, X=X_train)
                X_test = dt.process_molecular_features(pipeline=self.Pipeline, X=X_test)
                # train
                self.Train(X_train, y_train, hyperparameters, default_hyperparameters)

                y_pred_fold[test_idx] = self.Predict(X_test)

            y_pred_final += y_pred_fold
        y_pred_final /= n_outer
        return pd.Series(y_pred_final)
    
    def Get_Measure(self, X: pd.Series, y: pd.Series, hyperparameters: dict, n_outer: int, n_cv: int, 
                    threshold: float, goal: str) -> float:
        """
        Calculate the expected cross-validated measure (goal) for certain data

        Args:
            X (pd.Series): predictors/features
            y (pd.Series): target
            hyperparameters (dict): current hyperparameters
            n_outer (int): how many times the CV should be done
            n_cv (int): how many folds within the CV procedure
            threshold (float): the threshold for the target binarization (relevant to classifiers)
            goal (str): the chosen goal

        Returns:
            float: the value of a goal function (the quality measure of the predictive model)
        """        
        preds = self.Train_CV(hyperparameters, n_outer, n_cv)
        measures = self.Test_Model(y, preds, threshold)
        return measures[goal]

    def Get_All_Measures(self, X: pd.Series, y: pd.Series, hyperparameters: dict, n_outer: int, n_cv: int, 
                         threshold: float, default_hyperparameters=False) -> dict:
        """
        Calculate the expected cross-validated measures for certain data

        Args:
            X (pd.Series): predictors/features
            y (pd.Series): target
            hyperparameters (dict): current hyperparameters
            n_outer (int): how many times the CV should be done
            n_cv (int): how many folds within the CV procedure
            threshold (float): the threshold for the target binarization (relevant to classifiers)

        Returns:
            dict: the quality measures of the predictive model
        """        
        preds = self.Train_CV(hyperparameters, n_outer, n_cv, default_hyperparameters)
        measures = self.Test_Model(y, preds, threshold)
        return measures

    def Split_And_Train(self, X: pd.Series, y: pd.Series, hyperparameters: dict, random_state: int,
                        test_size: float) -> tuple:
        """
        Split the data and train

        Args:
            X (pd.Series): predictors/features
            y (pd.Series): target
            hyperparameters (dict): current hyperparameters
            random_state (int): random seed
            test_size (float): a fraction of the data treated as a test set

        Returns:
            tuple: the data after the split (the model is stored within the class)
        """        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.Train(X_train, y_train, hyperparameters)
        return X_train, X_test, y_train, y_test

    def F_Opt(self, space: dict, aux_data: dict) -> float:
        """
        The function needed for the hyperopt framework (hyperparameters optimization).
        Calculates the quality measure of the model for certain set of hyperparameters

        Args:
            space (dict): the current hyperparameters
            aux_data (dict): auxiliary information defining the calculation

        Returns:
            float: the optimization goal function
        """        
        hps = {}
        for key in space.keys(): hps[key] = self._hyperparameter_types[key](space[key])
        
        run_name = aux_data["run_name"]
        n_outer = aux_data["n_outer"]
        n_cv = aux_data["n_cv"]
        development = aux_data["development"]
        goal_function = aux_data["goal_function"]
        threshold = aux_data["threshold"]
        track_model = aux_data["track_model"]
        experiment = aux_data["experiment"]
        comment = aux_data["comment"]
        max_evals = aux_data["max_evals"]

        # Cross-validated measure based on the trainig set
        X = self._data["smiles_codes_train"]
        y = self._data["targets_train"]
        goal_function = self.Get_Measure(X, y, hps, n_outer, n_cv, threshold, goal_function)
        return aux_data["goal_function_multiplier"] * goal_function
    
    def Create_Features(self) -> NoReturn:
        """
        Calculate the untransformed molecular features.
        I.e. apply the first step of the pipeline 
        (turning the SMILES codes into the features
        according to the pipeline definition)

        Returns:
            NoReturn: the features are stored locally
        """        
        smiles_codes = self.Data["smiles_codes_train"]
        self._molecular_features = dt.create_molecular_features(pipeline=self.Pipeline, smiles_codes=smiles_codes)
           
    def Objective(self, params: dict, aux_data: dict) -> dict:
        """
        Wrapper for the F_Opt function

        Args:
            params (dict): the current hyperparameters
            aux_data (dict): auxiliary information defining the calculation

        Returns:
            dict: the output expected by the hyperopt framework
        """        
        output = self.F_Opt(params, aux_data)
        return {"loss": output ,  "status": STATUS_OK}
    
    def Load(self, path: str) -> NoReturn:
        """
        Load the previously stored predictive model into memory.
        It also creates the pipeline, and reads in the Keras model

        Args:
            path (str): path to the model

        Returns:
            NoReturn: the model is read into the memory
        """        
        
        with open(path, "rb") as f:
            self.__dict__ = pickle.load(f)
        self._pipeline = dt.create_pipeline(self._pipeline_configuration["feature_transform"])
        if self.__class__.__name__ == Prediction_Methodology.MLP:
            self._model = self._create_mlp_model(self._n_features)
            keras_model_name = Path(path)
            keras_model_name = keras_model_name.with_suffix(".model_weights")
            self._model.load_weights(keras_model_name)
            
    def Save(self, path: str) -> NoReturn:
        """
        Save the model on file.

        Args:
            path (str): path to the model

        Returns:
            NoReturn: 
        """        
        content = copy.copy(self.__dict__)
        # due to issues with mordred/rdkit modiles serialization - we remove the pipeline
        to_be_removed = ["_pipeline"]
        # the keras/tf models are not easily pickable
        if self.__class__.__name__ == Prediction_Methodology.MLP: to_be_removed += ["_model"]
        for item in to_be_removed:
            if item in content.keys(): content.pop(item)
        with open(path, "wb") as f:
            pickle.dump(content, f)
        if self.__class__.__name__ == Prediction_Methodology.MLP:
            keras_model_name = Path(path)
            keras_model_name = keras_model_name.with_suffix(".model_weights")
            self._model.save_weights(keras_model_name)
    
    @property
    def Methodology(self):
        return self._methodology
    
    @property
    def Space(self):
        return self._space
    
    @property
    def Model(self):
        return self._model

    @property
    def Validate(self):
        return self._validate    
    
    @Validate.setter
    def Validate(self, value):
        self._validate = value

    @property
    def Debug(self):
        return self._debug    
    
    @Debug.setter
    def Debug(self, value):
        self._debug = value
        
    @property
    def Data(self):
        return self._data
    
    @Data.setter
    def Data(self, value):
        self._data = value
        
    @property
    def Hyperparameters(self):
        return self._hyperparameters
    
    @Hyperparameters.setter
    def Hyperparameters(self, value):
        self._hyperparameters = value
        
    @property
    def Pipeline(self):
        return self._pipeline
    
    @Pipeline.setter
    def Pipeline(self, value):
        self._pipeline = value
        
    @property
    def OptimizationHistory(self):
        return self._optimization_history
    
    @OptimizationHistory.setter
    def OptimizationHistory(self, value):
        self._optimization_history = value
        
    @property
    def Validate(self):
        return self._validate
    
    @Validate.setter
    def Validate(self, value):
        self._validate = value
        
    @property
    def Measures(self):
        return self._measaures
    
    @Measures.setter
    def Measures(self, value):
        self._measaures = value

    @property
    def DefaultMeasures(self):
        return self._default_measures
    
    @DefaultMeasures.setter
    def DefaultMeasures(self, value):
        self._default_measures = value

    @property
    def DefaultHyperparameters(self):
        return self._default_hyperparameters
    
    @DefaultHyperparameters.setter
    def DefaultHyperparameters(self, value):
        self._default_hyperparameters = value

    @property
    def HyperoptSpace(self):
        return self._space
    
    @HyperoptSpace.setter
    def HyperoptSpace(self, value):
        self._space = value

    @property
    def ModelStatus(self):
        return self._model_status
    
    @ModelStatus.setter
    def ModelStatus(self, value):
        self._model_status = value
        
    @property
    def FeatureSet(self):
        return self._feature_set
    
    @FeatureSet.setter
    def FeatureSet(self, value):
        self._feature_set = value
  
    @property
    def GPUsupport(self):
        return self._GPU_support
    
    @GPUsupport.setter
    def GPUsupport(self, value):
        self._GPU_support = value

    @property
    def TrainingParams(self):
        return self._training_params
    
    @TrainingParams.setter
    def TrainingParams(self, value):
        self._training_params = value

    @property
    def PipelineConfiguration(self):
        return self._pipeline_configuration
    
    @PipelineConfiguration.setter
    def PipelineConfiguration(self, value):
        self._pipeline_configuration = value
        
    @abc.abstractclassmethod
    def Train(self, X: pd.DataFrame, y: pd.Series, hyperparameters: dict, default_hyperparameters: bool) -> Any:
        raise NotImplementedError
    
    @abc.abstractclassmethod
    def Predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

class M_XGBoost(Model):
    def __init__(self, prediction_type: Prediction_Type, pipeline_configuration: dict) -> NoReturn:
        """
        The M_XGBoost contructor

        Args:
            prediction_type (Prediction_Type): the class of predictive service, regression or classification
            pipeline_configuration (dict): the parametrization of the data creation/preprocessing

        Returns:
            NoReturn: 
        """        
        
        self._prediction_type = prediction_type
        
        self._hyperparameters = None
        self._hyperparameter_types = None
        self._tree_method = "auto"
        
        default_hyperparameters = {
            "n_estimators": 20,
            "max_depth": 20,
            "min_child_weight": 1,
            "eta": 0.3,
            "subsample": 0.3,
            "colsample_bytree": 1,
            "gamma": 0.0,
            "reg_alpha": 0,
            "reg_lambda": 1}
        
        if prediction_type == Prediction_Type.CLASSIFICATION:
            default_hyperparameters["objective"] = "reg:logistic"
        elif prediction_type == Prediction_Type.REGRESSION:
            default_hyperparameters["objective"] = "reg:squarederror"
        
        default_hyperparameter_types = {
            "n_estimators": int,
            "max_depth": int,
            "min_child_weight": float,
            "eta": float,
            "subsample": float,
            "colsample_bytree": float,
            "gamma": float,
            "reg_alpha": float,
            "reg_lambda": float,
            "objective": str            
        }
        
        default_space = {
            "max_depth": hp.quniform("max_depth", 3, 22, 1),
            "gamma": hp.uniform("gamma", 0, 9),
            "reg_alpha" : hp.uniform("reg_alpha", 0, 2),
            "reg_lambda" : hp.uniform("reg_lambda", 0, 2),
            "colsample_bytree" : hp.uniform("colsample_bytree", 0.6, 1.0),
            "min_child_weight" : hp.quniform("min_child_weight", 0, 10, 1),
            "n_estimators": hp.quniform("n_estimators", 100, 220, 1),
            "eta": hp.uniform("eta", 0.05, 1.0),
            "subsample": hp.uniform("subsample", 0.0, 1.0)
        }
        # default_space = {
        #     'max_depth': hp.quniform('max_depth', 3, 18, 1),
        #     'gamma': hp.uniform('gamma', 0, 9),
        #     'reg_alpha' : hp.uniform('reg_alpha', 0, 2),
        #     'reg_lambda' : hp.uniform('reg_lambda', 0, 2),
        #     'colsample_bytree' : hp.uniform('colsample_bytree', 0.6, 1.0),
        #     'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        #     'n_estimators': hp.quniform('n_estimators', 100, 180, 1),
        #     'eta': hp.uniform('eta', 0.05, 1.0),
        #     'subsample': hp.uniform('subsample', 0.0, 1.0)
        # }
        super().__init__(prediction_type, default_hyperparameters, default_hyperparameter_types, default_space, 
                         pipeline_configuration)
        
   
    def Train(self, X: pd.DataFrame, y: pd.Series, hyperparameters: dict, default_hyperparameters=False) -> NoReturn:
        """
        Train the model, potentially with validataion

        Args:
            X (pd.DataFrame): molecular descriptors/features
            y (pd.Series): targets (classes, numerical values)
            hyperparameters (dict): current hyperparameters

        Raises:
            NotImplementedError: only classification/regression is supported

        Returns:
            NoReturn: 
        """ 
        #self._refresh_hyperparameters(hyperparameters)
        # update the default hyperparameters
        current_hyperparameters = None
        if default_hyperparameters:
            current_hyperparameters = self._default_hyperparameters
        else:
            self._refresh_hyperparameters(hyperparameters)
            current_hyperparameters = self._hyperparameters
        
        if (self._validate):
            X_train, X_val, y_train, y_val = train_test_split(X, y.astype(int), test_size=0.10, random_state=42)
            eval_set = [(X_val, y_val)]
        else:
            X_train, y_train = X, y
            eval_set = None
        
        if self._prediction_type == Prediction_Type.CLASSIFICATION:
            self._model = XGBClassifier(use_label_encoder=False, eval_metric=["error", "logloss"], tree_method=self._tree_method, **current_hyperparameters)
            self._model.fit(X_train, y_train, eval_set=eval_set, verbose=0)
        elif self._prediction_type == Prediction_Type.REGRESSION:
            self._model = XGBRegressor(use_label_encoder=False, eval_metric=["rmse"], tree_method=self._tree_method, **current_hyperparameters)
            self._model.fit(X_train, y_train, eval_set=eval_set, verbose=0)
        else:
            raise NotImplementedError
        
        self._feature_set = X_train.columns
    
    
    def Predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Carries out the prediction
 
        Args:
            X (pd.DataFrame): molecular descriptors/features

        Returns:
            pd.Series: the predictions
        """        
        if self._model:
            X = X[self._feature_set]
            if self._prediction_type == Prediction_Type.CLASSIFICATION:
                preds = pd.Series(self._model.predict_proba(X)[:, 1], index=X.index)
            elif self._prediction_type == Prediction_Type.REGRESSION:
                preds = pd.Series(self._model.predict(X), index=X.index)
        else:
            preds = None
        return preds
    
    def Feature_Importance(self, importance_type="gain") -> dict:
        """
        Calculates the feature importance

        Args:
            importance_type (str, optional): Defines the type of the feature importance. Defaults to "gain".

        Returns:
            dict: the feature importance dictionary
        """        
        fi = self._model.get_booster().get_score(importance_type=importance_type)
        fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
        return fi

    @property
    def TreeMethod(self):
        return self._tree_method

    @TreeMethod.setter
    def TreeMethod(self, value):
        self._tree_method = value

class M_MLP(Model):
    def __init__(self, prediction_type: Prediction_Type, pipeline_configuration: dict) -> NoReturn:
        """
        The M_MLP constructor

        Args:
            prediction_type (Prediction_Type): the class of predictive service, regression or classification
            pipeline_configuration (dict): the parametrization of the data creation/preprocessing

        Returns:
            NoReturn:
        """        
        self._prediction_type = prediction_type
        
        self._hyperparameters = None
        self._hyperparameter_types = None
        self._verbose = 1
        self._n_features = None
        self._history = None
        
        default_hyperparameters = {
        "mlp_architecture": 1,
        "activation": 1,
        "batch_size": 64,
        "initial_lr": 0.001,
        "regularization_weight": 0.075,
        "optimizer": 3, 
        "min_lr": 0.001,
        "dropout_rate": 0.1
        }
        
        default_hyperparameter_types = {
            "mlp_architecture": lambda x: x,
            "activation": str,
            "batch_size": int,
            "initial_lr": float,
            "regularization_weight": float,
            "optimizer": str,
            "min_lr": float,
            "dropout_rate": float
        }
        
        default_space = {
            "mlp_architecture": hp.choice("mlp_architecture", [[64, 32], [128, 64], [256, 128], [512, 256],  [64, 32, 32], [64, 32, 16], 
                                                               [128, 64, 32], [128, 128, 64], [256, 128, 64], [512, 256, 128]]),
            "activation": hp.choice("activation", ["relu", "selu", "elu", "tanh"]),
            "batch_size": hp.choice("batch_size", [8, 16, 32, 64, 96, 128, 256]),
            "initial_lr": hp.uniform("initial_lr", 0.01, 0.02),
            "regularization_weight": hp.uniform("regularization_weight", 0.01, 0.1),
            "optimizer": hp.choice("optimizer", ["SGD", "RMSprop", "Adagrad", "Adam", "Adamax", "Nadam"]),
            "min_lr": hp.uniform("min_lr", 0.00001, 0.001),
            "dropout_rate": hp.uniform("dropout_rate", 0.0, 0.5),
        }
        super().__init__(prediction_type, default_hyperparameters, default_hyperparameter_types, default_space, 
                         pipeline_configuration)
        
    def _create_mlp_model(self, n_features: int) -> Sequential:
        """
        Prepares the Keras model, the multilayer perceptron with
        the architecture defined in hyperparameters

        Args:
            hyperparameters (dict): the hyperparameters of the model
            n_features (int): the number of features 
            (i.e. the number of neuraons on the first layer)

        Raises:
            NotImplementedError:

        Returns:
            Sequential: the resulting model
        """        
        
        mlp_architecture = self._hyperparameters["mlp_architecture"]
        regularization_weight = self._hyperparameters["regularization_weight"]
        activation = self._hyperparameters["activation"]
        optimizer = self._hyperparameters["optimizer"]
        dropout_rate = self._hyperparameters["dropout_rate"]
        
        model = Sequential(name="mlp_model")
        model.add(InputLayer(input_shape=(n_features,), name="input_layer"))
        for idx, dim in enumerate(mlp_architecture):
            model.add(Dense(dim, activation=activation, activity_regularizer=l1_l2(l1=regularization_weight, l2=regularization_weight),
                            #kernel_initializer=initializers.RandomNormal(stddev=0.01, seed=self._random_seed),
                            #bias_initializer=initializers.Zeros(), 
                            name="dense_"+str(idx)))
            model.add(Dropout(dropout_rate, name="dropout_"+str(idx)))
        # TODO final activation can also be a subject of optimization
        if self._prediction_type == Prediction_Type.CLASSIFICATION:
            final_activation = "sigmoid"
        elif self._prediction_type == Prediction_Type.REGRESSION:
            final_activation = activation
        else:
            raise NotImplementedError
            
        model.add(Dense(1, activation=final_activation, name="output_layer"
                        #kernel_initializer=initializers.RandomNormal(stddev=0.01, seed=self._random_seed),
                        #bias_initializer=initializers.Zeros(), 
                        ))

        #bc = metrics.BinaryCrossentropy(name="binary_crossentropy", dtype=None, from_logits=False, label_smoothing=0)

        mtrcs = ["accuracy"]#, bc]
        learning_rate = self._hyperparameters["initial_lr"]
        if optimizer == "Adam":
            o = Adam(learning_rate=learning_rate)
        elif optimizer == "SGD":
            o = SGD(learning_rate=learning_rate)
        elif optimizer == "RMSprop":
            o = RMSprop(learning_rate=learning_rate)
        elif optimizer == "Adagrad":
            o = Adagrad(learning_rate=learning_rate)
        elif optimizer == "Adadelta":
            o = Adadelta(learning_rate=learning_rate)
        elif optimizer == "Adamax":
            o = Adamax(learning_rate=learning_rate)
        elif optimizer == "Nadam":
            o = Nadam(learning_rate=learning_rate)

        if self._prediction_type == Prediction_Type.CLASSIFICATION:
            loss_function = "binary_crossentropy"
        elif self._prediction_type == Prediction_Type.REGRESSION:
            loss_function = "mse"

        model.compile(optimizer=o, loss=loss_function, metrics=mtrcs)
        return model
    
    def Train(self, X: pd.DataFrame, y: pd.Series, hyperparameters: dict, default_hyperparameters=False) -> keras.callbacks.History:
        """
        Train the model, potentially with validataion
        
        Args:
            X (pd.DataFrame): molecular descriptors/features
            y (pd.Series): targets (classes, numerical values)
            hyperparameters (dict): current hyperparameters

        Returns:
            keras.callbacks.History: history of the training
        """
        # update the default hyperparameters
        current_hyperparameters = None
        if default_hyperparameters:
            current_hyperparameters = self._default_hyperparameters
        else:
            self._refresh_hyperparameters(hyperparameters)
            current_hyperparameters = self._hyperparameters
                
        if (self._validate):
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)
            eval_set = (X_val, y_val)
        else:
            X_train, y_train = X, y
            eval_set = None
        
        self._n_features = X_train.shape[1]
        self._model = self._create_mlp_model(self._n_features)
        
        checkpoint_file = tempfile.NamedTemporaryFile()
        
        if self._validate:
            monitor_metrics = "val_loss"
        else:
            monitor_metrics = "loss"
        
        reduce_lr = ReduceLROnPlateau(monitor=monitor_metrics, factor=0.1, patience=5, 
                                      min_lr=self._hyperparameters["min_lr"], verbose=10)
        early_stopping = EarlyStopping(monitor=monitor_metrics, mode="auto", patience=7, verbose=10, min_delta=0.001)
        model_checkpoint = ModelCheckpoint(checkpoint_file.name, monitor="val_loss", verbose=10, 
                                           save_best_only=True, save_weights_only=True, mode ="min")

        callbacks = [reduce_lr, early_stopping, model_checkpoint]

        batch_size = self._hyperparameters["batch_size"]
        max_epochs = 4096
        result = self._model.fit(X_train, y_train, epochs=max_epochs, batch_size=batch_size, verbose=self._verbose,
                                 validation_data=eval_set, callbacks=callbacks)
        try:
            self._model.load_weights(checkpoint_file.name)
        except OSError:
            l.info("Issue with the checkpoint file...")
            
        checkpoint_file.close()
        return result
        
    def Predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Carries out the prediction
 
        Args:
            X (pd.DataFrame): molecular descriptors/features

        Returns:
            pd.Series: the predictions
        """       
        if self._model:
            preds = pd.Series(self._model.predict(X).flatten(), index=X.index)
        else:
            preds = None
        return preds
    
    def Plot_Train_History(self, history: dict) -> plt.Axes.axes:
        """
        Create the plot reflecting the training evolution

        Args:
            history (keras.callbacks.History): history of the training

        Returns:
            plt.Axes.axes: matplotlib plot object
        """
        f, ax = plt.subplots(figsize=(10, 10))
        ax.plot(history["loss"], label="training")
        ax.plot(history["val_loss"], label="validation")
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid()
        return ax

    def Force_CPU(self) -> NoReturn:
        """
        Force the calculation to be done on CPU

        Returns:
            NoReturn: _description_
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    @property
    def Verbose(self):
        return self._verbose

    @Verbose.setter
    def Verbose(self, value):
        self._verbose = value
        
    @property
    def History(self):
        return self._history

    @History.setter
    def History(self, value):
        self._history = value

class M_RandomForest(Model):
    def __init__(self, prediction_type: Prediction_Type, pipeline_configuration: dict) -> NoReturn:
        """
        The M_RandomForest constructor

        Args:
            prediction_type (Prediction_Type): the class of predictive service, regression or classification
            pipeline_configuration (dict): the parametrization of the data creation/preprocessing

        Returns:
            NoReturn:
        """
        self._prediction_type = prediction_type
        
        self._hyperparameters = None
        self._hyperparameter_types = None
        if prediction_type == Prediction_Type.CLASSIFICATION:      
            default_hyperparameters = {
                "bootstrap": True,
                "ccp_alpha": 0.0,
                "criterion": "gini",
                "max_depth": 10,
                "min_impurity_decrease": 0.0,
                "min_samples_leaf": 1,
                "min_samples_split": 2,
                "min_weight_fraction_leaf": 0.0,
                "n_estimators": 100,
                "n_jobs": co.N_JOBS,
                "oob_score": False,
                "random_state": co.RANDOM_STATE,
                "verbose": 0,
                "warm_start": False
            }
            default_hyperparameter_types = {
                "bootstrap": bool,
                "ccp_alpha": float,
                "criterion": str,
                "max_depth": int,
                "min_impurity_decrease": float,
                "min_samples_leaf": int,
                "min_samples_split": int,
                "min_weight_fraction_leaf": float,
                "n_estimators": int,
                "n_jobs": int,
                "oob_score": bool,
                "random_state": int,
                "verbose": int,
                "warm_start": bool
            }
            default_space = {
                #"bootstrap": hp.choice("bootstrap", [True, False]),
                #"ccp_alpha": hp.uniform("ccp_alpha", 0.0, 0.1),
                "criterion": hp.choice("criterion", ["gini", "entropy", "log_loss"]),
                "max_depth": hp.quniform("max_depth", 20, 50, 2),
                #"min_impurity_decrease": hp.uniform("min_impurity_decrease", 0.0, 0.1),
                "n_estimators": hp.quniform("n_estimators", 80, 200, 2),
                
            }
        elif prediction_type == Prediction_Type.REGRESSION:
            default_hyperparameters = {
                "bootstrap": True,
                "ccp_alpha": 0.0,
                #"class_weight": None,
                "criterion": "squared_error",
                "max_depth": 10,
                #"max_leaf_nodes": None,
                #"max_samples": None,
                "min_impurity_decrease": 0.0,
                #"min_impurity_split": None,
                "min_samples_leaf": 1,
                "min_samples_split": 2,
                "min_weight_fraction_leaf": 0.0,
                "n_estimators": 100,
                "n_jobs": co.N_JOBS,
                "oob_score": False,
                #"random_state": None,
                "verbose": 0,
                "warm_start": False
            }
            default_hyperparameter_types = {
                "bootstrap": bool,
                "ccp_alpha": float,
                "criterion": str,
                "max_depth": int,
                "min_impurity_decrease": float,
                "min_samples_leaf": int,
                "min_samples_split": int,
                "min_weight_fraction_leaf": float,
                "n_estimators": int,
                "n_jobs": int,
                "oob_score": bool,
                "verbose": int,
                "warm_start": bool
            }
            default_space = {
                #"criterion": hp.choice("criterion", ["squared_error", "absolute_error", "poisson"]),
                "max_depth": hp.quniform("max_depth", 20, 60, 2),
                #"min_impurity_decrease": hp.uniform("min_impurity_decrease", 0.0, 0.1),
                "n_estimators": hp.quniform("n_estimators", 80, 200, 2),                
            }
        super().__init__(prediction_type, default_hyperparameters, default_hyperparameter_types, default_space, 
                         pipeline_configuration)
        
    def Train(self, X: pd.DataFrame, y: pd.Series, hyperparameters: dict, default_hyperparameters=False) -> NoReturn:
        """
        Train the model, potentially with validataion
        
        Args:
            X (pd.DataFrame): molecular descriptors/features
            y (pd.Series): targets (classes, numerical values)
            hyperparameters (dict): current hyperparameters        

        Raises:
            NotImplementedError: 

        Returns:
            NoReturn:
        """
        # update the default hyperparameters
        current_hyperparameters = None
        if default_hyperparameters:
            current_hyperparameters = self._default_hyperparameters
        else:
            self._refresh_hyperparameters(hyperparameters)
            current_hyperparameters = self._hyperparameters
        
        if self._prediction_type == Prediction_Type.CLASSIFICATION:
            self._model = RandomForestClassifier(**self._hyperparameters)
        elif self._prediction_type == Prediction_Type.REGRESSION:
            self._model = RandomForestRegressor(**self._hyperparameters)
        else:
            raise NotImplementedError
        
        self._model.fit(X, y)

    def Predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Carries out the prediction
 
        Args:
            X (pd.DataFrame): molecular descriptors/features

        Returns:
            pd.Series: the predictions
        """       
        if self._model:
            # preds = pd.Series(self._model.predict(X))
            if self._prediction_type == Prediction_Type.CLASSIFICATION:
                preds = pd.Series(self._model.predict_proba(X)[:, 1], index=X.index)
            elif self._prediction_type == Prediction_Type.REGRESSION:
                preds = pd.Series(self._model.predict(X), index=X.index)
        else:
            preds = None
        return preds

class M_Bagging(Model):
    def __init__(self, prediction_type: Prediction_Type, pipeline_configuration: dict) -> NoReturn:
        """
        The M_Bagging constructor

        Args:
            prediction_type (Prediction_Type): the class of predictive service, regression or classification
            pipeline_configuration (dict): the parametrization of the data creation/preprocessing

        Returns:
            NoReturn:
        """
        self._prediction_type = prediction_type
        
        self._hyperparameters = None
        self._hyperparameter_types = None

        default_hyperparameters = {
            #"base_estimator": None,
            "bootstrap": True,
            "bootstrap_features": False,
            "max_features": 1.0,
            "max_samples": 1.0,
            "n_estimators": 10,
            "oob_score": False,
            #"random_state": None,
            "verbose": 0,
            "warm_start": False,
            "n_jobs": 8
        }
        
        default_hyperparameter_types = {
            "bootstrap": bool,
            "bootstrap_features": bool,
            "max_features": float,
            "max_samples": float,
            "n_estimators": int,
            "oob_score": bool,
            "verbose": int,            
            "warm_start": bool,
            "n_jobs": int
        }
        
        default_space={
            "n_estimators": hp.quniform("n_estimators", 1, 75, 1),
            "max_features": hp.uniform("max_features", 0.75, 1.0),
            "max_samples": hp.uniform("max_samples", 0.75, 1.0)
       }
        super().__init__(prediction_type, default_hyperparameters, default_hyperparameter_types, default_space, 
                         pipeline_configuration)
   
    def Train(self, X: pd.DataFrame, y: pd.Series, hyperparameters: dict, default_hyperparameters=False) -> NoReturn:
        """
        Train the model, potentially with validataion
        
        Args:
            X (pd.DataFrame): molecular descriptors/features
            y (pd.Series): targets (classes, numerical values)
            hyperparameters (dict): current hyperparameters        

        Raises:
            NotImplementedError: 

        Returns:
            NoReturn:
        """
        # update the default hyperparameters
        current_hyperparameters = None
        if default_hyperparameters:
            current_hyperparameters = self._default_hyperparameters
        else:
            self._refresh_hyperparameters(hyperparameters)
            current_hyperparameters = self._hyperparameters
    
        if (self._validate):
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)
            eval_set = [(X_val, y_val)]
        else:
            X_train, y_train = X, y
            eval_set = None

        if self._prediction_type == Prediction_Type.CLASSIFICATION:
            self._model = BaggingClassifier(**self._hyperparameters)
        elif self._prediction_type == Prediction_Type.REGRESSION:
            self._model = BaggingRegressor(**self._hyperparameters)
        else:
            raise NotImplementedError

        self._model.fit(X_train, y_train)
    
    def Predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Carries out the prediction
 
        Args:
            X (pd.DataFrame): molecular descriptors/features

        Returns:
            pd.Series: the predictions
        """       
        if self._model:
            if self._prediction_type == Prediction_Type.CLASSIFICATION:
                preds = pd.Series(self._model.predict_proba(X)[:, 1], index=X.index)
            elif self._prediction_type == Prediction_Type.REGRESSION:
                preds = pd.Series(self._model.predict(X), index=X.index)
        else:
            preds = None
        return preds
    
    def Coeff_Histogram(self, bins=50):
        """
        Create the linear coefficionets distribution

        Args:
            bins (_type_, optional): Number of histogram bins

        Returns:
            plt.figure: the matplotlib figure object
        """
        coeffs = self._model.coef_
        plt.hist(coeffs[0, :], bins=bins)
        plt.grid(True)
        
        return plt.figure

class M_SVM(Model):
    def __init__(self, prediction_type: Prediction_Type, pipeline_configuration: dict) -> NoReturn:
        """
        The M_SVM constructor

        Args:
            prediction_type (Prediction_Type): the class of predictive service, regression or classification
            pipeline_configuration (dict): the parametrization of the data creation/preprocessing

        Returns:
            NoReturn:
        """
        self._prediction_type = prediction_type
        
        self._hyperparameters = None
        self._hyperparameter_types = None
        
        if prediction_type == Prediction_Type.CLASSIFICATION:
            default_hyperparameters = {    
                "C": 2.0,
                "break_ties": False,
                "cache_size": 200,
                #"class_weight": None,
                "coef0": 0.0,
                "decision_function_shape": "ovr",
                "degree": 3,
                "gamma": "scale",
                "kernel": "rbf",
                "max_iter": -1,
                "probability": True,
                #"random_state": None,
                "shrinking": True,
                "tol": 0.001,
                "verbose": False
            }
            
            default_hyperparameter_types = {
                "C": float,
                "kernel": str,
                "break_ties": bool,
                "cache_size": int,
                "coef0": float,
                "decision_function_shape": str,
                "degree": int,
                "gamma": str,
                "kernel": str,
                "max_iter": int,
                "probability": bool,
                "shrinking": bool,
                "tol": float,
                "verbose": bool
            }
            
            default_space={
                "C": hp.uniform("C", 0.1, 2.0),
                "kernel": hp.choice("kernel", ["linear", "poly", "rbf", "sigmoid"]),
                "shrinking": hp.choice("shrinking", [True, False]),
            }
        elif prediction_type == Prediction_Type.REGRESSION:
            default_hyperparameters = {    
                "C": 2.0,
                #"break_ties": False,
                "cache_size": 200,
                #"class_weight": None,
                "coef0": 0.0,
                #"decision_function_shape": "ovr",
                "degree": 3,
                "gamma": "scale",
                "kernel": "rbf",
                "max_iter": -1,
                #"probability": True,
                #"random_state": None,
                "epsilon": 0.1,
                "shrinking": True,
                "tol": 0.001,
                "verbose": False
            }
            
            default_hyperparameter_types = {
                "C": float,
                "kernel": str,
                "break_ties": bool,
                "cache_size": int,
                "coef0": float,
                "decision_function_shape": str,
                "degree": int,
                "gamma": str,
                "kernel": str,
                "max_iter": int,
                #"probability": bool,
                "epsilon": float,
                "shrinking": bool,
                "tol": float,
                "verbose": bool
            }
            
            default_space={
                "C": hp.uniform("C", 0.1, 10),
                "kernel": hp.choice("kernel", ["linear", "poly", "rbf", "sigmoid"]),
                "shrinking": hp.choice("shrinking", [True, False]),
                "epsilon": hp.uniform("epsilon", 0.05, 1.0)
            }            
        super().__init__(prediction_type, default_hyperparameters, default_hyperparameter_types, default_space, 
                         pipeline_configuration)
   
    def Train(self, X: pd.DataFrame, y: pd.Series, hyperparameters: dict, default_hyperparameters=False) -> NoReturn:
        """
        Train the model, potentially with validataion
        
        Args:
            X (pd.DataFrame): molecular descriptors/features
            y (pd.Series): targets (classes, numerical values)
            hyperparameters (dict): current hyperparameters        

        Raises:
            NotImplementedError: 

        Returns:
            NoReturn:
        """
        # update the default hyperparameters
        current_hyperparameters = None
        if default_hyperparameters:
            current_hyperparameters = self._default_hyperparameters
        else:
            self._refresh_hyperparameters(hyperparameters)
            current_hyperparameters = self._hyperparameters
    
        if (self._validate):
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)
            eval_set = [(X_val, y_val)]
        else:
            X_train, y_train = X, y
            eval_set = None
            
        if self._prediction_type == Prediction_Type.CLASSIFICATION:
            self._model = SVC(**self._hyperparameters)
        elif self._prediction_type == Prediction_Type.REGRESSION:
            self._model = SVR(**self._hyperparameters)

        self._model.fit(X_train, y_train)
    
    def Predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Carries out the prediction
 
        Args:
            X (pd.DataFrame): molecular descriptors/features

        Returns:
            pd.Series: the predictions
        """       
        if self._model:
            if self._prediction_type == Prediction_Type.CLASSIFICATION:
                if self._hyperparameters["probability"]:
                    preds = pd.Series(self._model.predict_proba(X)[:, 1], index=X.index)
                else:
                    preds = pd.Series(self._model.predict(X))
            elif self._prediction_type == Prediction_Type.REGRESSION:
                preds = pd.Series(self._model.predict(X), index=X.index)
        else:
            preds = None
        return preds

class M_Ridge(Model):
    def __init__(self, prediction_type: Prediction_Type, pipeline_configuration: dict) -> NoReturn:
        """
        The M_Ridge constructor

        Args:
            prediction_type (Prediction_Type): the class of predictive service, regression or classification
            pipeline_configuration (dict): the parametrization of the data creation/preprocessing

        Returns:
            NoReturn:
        """
        self._prediction_type = prediction_type
        
        self._hyperparameters = None
        self._hyperparameter_types = None
        
        if prediction_type == Prediction_Type.CLASSIFICATION:
            default_hyperparameters = {
                "alpha": 1.0,
                "fit_intercept": True,
                "max_iter": 15000,
                "tol": 0.001,
                "solver": "auto"
                }

            default_hyperparameter_types = {
                "alpha": float,
                "fit_intercept": bool,
                "max_iter": int,
                "tol": float,
                "solver": str
                }
            default_space = {
                "alpha": hp.uniform("alpha", 0.1, 5.0),
                "solver": hp.choice("solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])
            }
        elif prediction_type == Prediction_Type.REGRESSION:
            default_hyperparameters = {
                "alpha": 1.0,
                "fit_intercept": True,
                "max_iter": 15000,
                "tol": 0.001,
                "solver": "auto"
                }

            default_hyperparameter_types = {
                "alpha": float,
                "fit_intercept": bool,
                "max_iter": int,
                "tol": float,
                "solver": str
                }
            default_space = {
                "alpha": hp.uniform("alpha", 0.1, 5.0),
                "solver": hp.choice("solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])
            }
        super().__init__(prediction_type, default_hyperparameters, default_hyperparameter_types, default_space, 
                         pipeline_configuration)
   
    def Train(self, X: pd.DataFrame, y: pd.DataFrame, hyperparameters: dict, default_hyperparameters=False) -> int:
        """
        Train the model, potentially with validataion
        
        Args:
            X (pd.DataFrame): molecular descriptors/features
            y (pd.Series): targets (classes, numerical values)
            hyperparameters (dict): current hyperparameters        

        Raises:
            NotImplementedError: 

        Returns:
            NoReturn:
        """
        # update the default hyperparameters
        current_hyperparameters = None
        if default_hyperparameters:
            current_hyperparameters = self._default_hyperparameters
        else:
            self._refresh_hyperparameters(hyperparameters)
            current_hyperparameters = self._hyperparameters
    
        if (self._validate):
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)
            eval_set = [(X_val, y_val)]
        else:
            X_train, y_train = X, y
            eval_set = None

        if self._prediction_type == Prediction_Type.CLASSIFICATION:
            self._model = RidgeClassifier(**self._hyperparameters) #, index=X.index
        elif self._prediction_type == Prediction_Type.REGRESSION:
            self._model = Ridge(**self._hyperparameters) #, index=X.index #doesn't work for HIV.csv
        
        self._model.fit(X_train, y_train)
    
    def Predict(self, X: pd.Series) -> pd.Series:
        """
        Carries out the prediction
 
        Args:
            X (pd.DataFrame): molecular descriptors/features

        Returns:
            pd.Series: the predictions
        """       
        if self._model:
            preds = pd.Series(self._model.predict(X))
        else:
            preds = None
        return preds
    
    def Coeff_Histogram(self, bins=50) -> plt.figure:
        """
        Create the linear coefficionets distribution

        Args:
            bins (_type_, optional): Number of histogram bins

        Returns:
            plt.figure: the matplotlib figure object
        """
        coeffs = self._model.coef_
        plt.hist(coeffs[0, :], bins=bins)
        plt.grid(True)
        
        return plt.figure