from pathlib import Path
import sys
import pandas as pd
import numpy as np
local_module_path = Path("../modules/model_wrapper")
sys.path.append(local_module_path.as_posix())
local_module_path = Path("../modules/data_transformers")
sys.path.append(local_module_path.as_posix())
local_module_path = Path("../modules/auxiliary")
sys.path.append(local_module_path.as_posix())
import data_transformers as dt
import constants as co

ARTIFACTS_DIR = Path(__file__).parent/"artifacts"

def test_pipeline_regression():

    smiles_codes = ["O=C(O)C(S)C(S)C(=O)O", "O=S(=O)(c1ccccc1)N1CCc2cc(C(O)(C(F)(F)F)C(F)(F)F)ccc21"]
    target_values = [31900, 883]
    
    smiles_codes = pd.Series(smiles_codes)
    target = pd.Series(target_values)
    
    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)
    
    pipeline_file = "test_pipeline_configuration_regression.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    fp_columns = pipeline_train["Molecular features"].fp_column_names
    assert X_train[fp_columns].sum().sum() == 52
    assert np.abs(y_train.sum() - 4.126338907446342) < co.EPSILON
    assert np.abs(X_train["QED"].sum() - 1.216724054855354) < co.EPSILON

def test_pipeline_regression_pca_n_components():

    test_data = pd.read_csv((ARTIFACTS_DIR/"test_data_1.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data["smiles"]
    target = test_data["target"]
    
    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)
    
    pipeline_file = "test_pipeline_configuration_regression_pca_n_components.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    pca_columns = pipeline_train["PCA"].PCA_feature_names
    assert X_train[pca_columns].sum().sum() < co.EPSILON * 5.0
    assert np.abs(y_train.sum() - 332.9964572007816) < co.EPSILON
    pca = pipeline_train["PCA"].PCA_model
    variance_explained = np.sum(pca.explained_variance_ratio_)
    assert np.abs(variance_explained - 0.5677026318621108) < co.EPSILON
    

def test_pipeline_regression_pca():

    test_data = pd.read_csv((ARTIFACTS_DIR/"test_data_1.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data["smiles"]
    target = test_data["target"]
    
    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)
    
    pipeline_file = "test_pipeline_configuration_regression_pca.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    pca_columns = pipeline_train["PCA"].PCA_feature_names
    assert X_train[pca_columns].sum().sum() < co.EPSILON * 10.0
    assert np.abs(y_train.sum() - 332.9964572007816) < co.EPSILON
    assert np.abs(X_train["QED"].sum() - 54.39274167788612) < co.EPSILON

def test_pipeline_regression_only_fps_morgan():

    test_data = pd.read_csv((ARTIFACTS_DIR/"test_data_1.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data["smiles"]
    target = test_data["target"]

    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)
    
    pipeline_file = "test_pipeline_configuration_regression_only_fps_morgan.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    fp_columns = pipeline_train["Molecular features"].fp_column_names
    assert X_train[fp_columns].sum().sum() == 4162
    assert np.abs(y_train.sum() - 333.1783870141875) < co.EPSILON
    
def test_pipeline_regression_only_fps_atompairs():

    test_data = pd.read_csv((ARTIFACTS_DIR/"test_data_1.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data["smiles"]
    target = test_data["target"]

    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)
    
    pipeline_file = "test_pipeline_configuration_regression_only_fps_morgan.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    fp_columns = pipeline_train["Molecular features"].fp_column_names
    assert X_train[fp_columns].sum().sum() == 4162
    assert np.abs(y_train.sum() - 333.1783870141875) < co.EPSILON

def test_pipeline_corr():

    test_data = pd.read_csv((ARTIFACTS_DIR/"test_data_1.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data["smiles"]
    target = test_data["target"]

    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)
    
    pipeline_file = "test_pipeline_configuration_regression_pca_scaling_corr.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    survived_features = pipeline_train["Corr"].SurvivedFeatures
    assert np.abs(X_train[survived_features].sum().sum() - 3877.0) < co.EPSILON

def test_pipeline_regression_only_fps_sheridan():

    test_data = pd.read_csv((ARTIFACTS_DIR/"test_data_1.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data["smiles"]
    target = test_data["target"]

    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)
    
    pipeline_file = "test_pipeline_configuration_regression_only_fps_morgan.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    fp_columns = pipeline_train["Molecular features"].fp_column_names
    assert X_train[fp_columns].sum().sum() == 4162
    assert np.abs(y_train.sum() - 333.1783870141875) < co.EPSILON
    
def test_pipeline_regression_only_fps_topological():

    test_data = pd.read_csv((ARTIFACTS_DIR/"test_data_1.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data["smiles"]
    target = test_data["target"]

    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)
    
    pipeline_file = "test_pipeline_configuration_regression_only_fps_topological.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    fp_columns = pipeline_train["Molecular features"].fp_column_names
    assert X_train[fp_columns].sum().sum() == 55726
    assert np.abs(y_train.sum() - 333.1783870141875) < co.EPSILON
    
def test_pipeline_regression_only_fps_maccs():

    test_data = pd.read_csv((ARTIFACTS_DIR/"test_data_1.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data["smiles"]
    target = test_data["target"]

    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)
    
    pipeline_file = "test_pipeline_configuration_regression_only_fps_maccs.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    fp_columns = pipeline_train["Molecular features"].fp_column_names
    assert X_train[fp_columns].sum().sum() == 4786
    assert np.abs(y_train.sum() - 333.1783870141875) < co.EPSILON

def test_pipeline_regression_only_descriptors():

    test_data = pd.read_csv((ARTIFACTS_DIR/"test_data_1.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data["smiles"]
    target = test_data["target"]
    
    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)
    
    pipeline_file = "test_pipeline_configuration_regression_only_descriptors.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    assert np.abs(y_train.sum() - 333.1783870141875) < co.EPSILON
    assert np.abs(X_train["QED"].sum() - 54.92231487365173) < co.EPSILON
    assert np.abs(X_train["SLogP"].sum() - 353.42059000000023) < co.EPSILON
    assert np.abs(X_train["MW"].sum() - 35544.69200639604) < co.EPSILON

def test_pipeline_regression_supplement():

    test_data = pd.read_csv((ARTIFACTS_DIR/"test_data_1.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data["smiles"]
    target = test_data["target"]
    
    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)
    
    pipeline_file = "test_pipeline_configuration_regression_supplement.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    assert "supp_1" in X_train.columns
    assert "supp_2" in X_train.columns
    assert X_train["supp_1"].sum() == 95
    assert X_train["supp_2"].sum() == 190

def test_pipeline_regression_scaling():
    
    smiles_codes = ["O=C(O)C(S)C(S)C(=O)O", "O=S(=O)(c1ccccc1)N1CCc2cc(C(O)(C(F)(F)F)C(F)(F)F)ccc21", "N#Cc1cccc(NC(=O)Nc2ccc3c(c2)C(N2CC4CCC(C2)N4C(=O)C2CCCC2)CC3)c1"]
    target_values = [31900, 883, 357.7]
    
    smiles_codes = pd.Series(smiles_codes)
    target = pd.Series(target_values)

    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)
    
    pipeline_file = "test_pipeline_configuration_regression_scaling.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    fp_columns = pipeline_train["Molecular features"].fp_column_names
    assert X_train[fp_columns].sum().sum() == 110
    assert np.abs(y_train.sum() - 4.4321309990704405) < co.EPSILON
    assert np.abs(X_train["QED"].sum()) < co.EPSILON
    assert np.abs(X_train["SLogP"].sum()) < co.EPSILON
    assert np.abs(X_train["MW"].sum()) < co.EPSILON
    
def test_pipeline_regression_scaling_more_classes():
    
    test_data = pd.read_csv((ARTIFACTS_DIR/"test_data_1.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data["smiles"]
    target = test_data["target"]

    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)
    
    pipeline_file = "test_pipeline_configuration_regression_scaling_more_classes.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    fp_columns = pipeline_train["Molecular features"].fp_column_names
    assert X_train[fp_columns].sum().sum() == 4123
    assert np.abs(y_train.sum() - 332.9964572007816) < co.EPSILON
    assert np.abs(X_train["QED"].sum()) < co.EPSILON*10.0
    assert np.abs(X_train["SLogP"].sum()) < co.EPSILON*10.0
    assert np.abs(X_train["MW"].sum()) < co.EPSILON*10.0
    assert np.abs(X_train["BalabanJ"].sum() - 153.77295010626483) < co.EPSILON

def test_pipeline_classification():

    smiles_codes = ["O=C(O)C(S)C(S)C(=O)O", "O=S(=O)(c1ccccc1)N1CCc2cc(C(O)(C(F)(F)F)C(F)(F)F)ccc21"]
    target_values = [31900, 883]
    
    smiles_codes = pd.Series(smiles_codes)
    target = pd.Series(target_values)
    
    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)
    
    pipeline_file = "test_pipeline_configuration_classification.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)

    fp_columns = pipeline_train["Molecular features"].fp_column_names
    assert X_train[fp_columns].sum().sum() == 52
    assert y_train.astype(np.int8).sum() == 1
    assert np.abs(X_train["QED"].sum() - 1.216724054855354) < co.EPSILON
    
def test_pipeline_regression_no_o_removal():

    smiles_codes = ["O=C(O)C(S)C(S)C(=O)O", "O=S(=O)(c1ccccc1)N1CCc2cc(C(O)(C(F)(F)F)C(F)(F)F)ccc21"]
    target_values = [31900, 883]
    
    smiles_codes = pd.Series(smiles_codes)
    target = pd.Series(target_values)
    
    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)
    
    pipeline_file = "test_pipeline_configuration_regression_no_o_removal.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    fp_columns = pipeline_train["Molecular features"].fp_column_names
    assert X_train[fp_columns].sum().sum() == 52
    assert np.abs(y_train.sum() - 4.126338907446342) < co.EPSILON
    assert np.abs(X_train["QED"].sum() - 1.216724054855354) < co.EPSILON

    
def test_XGBoost_wrapper_with_scaling():
    
    test_data = pd.read_csv((ARTIFACTS_DIR/"test_data_1.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data["smiles"]
    target = test_data["target"]

    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)
    
    pipeline_file = "test_pipeline_configuration_regression_scaling.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    fp_columns = pipeline_train["Molecular features"].fp_column_names
    assert X_train.shape == (95, 1035)
    assert X_train[fp_columns].sum().sum() == 4123
    assert np.abs(y_train.sum() - 332.9964572007816) < co.EPSILON
    assert np.abs(X_train["QED"].sum()) < co.EPSILON*100
    assert np.abs(X_train["SLogP"].sum()) < co.EPSILON*100
    assert np.abs(X_train["MW"].sum()) < co.EPSILON*100
    
#    m_r = mw.M_XGBoost(prediction_methodology=mw.Prediction_Methodology.XGBOOST, 
#                       prediction_type=mw.Prediction_Type.REGRESSION, pipeline=pipeline_train)

def test_pipeline_regression_fit_transform():

    test_data_fit = pd.read_csv((ARTIFACTS_DIR/"test_data_1.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data_fit["smiles"]
    target = test_data_fit["target"]

    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)
    
    pipeline_file = "test_pipeline_configuration_regression.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    # fitting
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    fp_columns = pipeline_train["Molecular features"].fp_column_names
    assert X_train[fp_columns].sum().sum() == 4123
    assert np.abs(y_train.sum() - 332.9964572007816) < co.EPSILON
    
    test_data_transform = pd.read_csv((ARTIFACTS_DIR/"test_data_2.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data_transform["smiles"]
    target = test_data_transform["target"]
    
    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)
    
    # just transforming with other/any data
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]    

    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    assert X_train[fp_columns].sum().sum() == 4854
    assert np.abs(X_train["QED"].sum() - 58.70206232985786) < co.EPSILON
    assert np.abs(y_train.sum() - 121.30807004481103) < co.EPSILON
    
    
    