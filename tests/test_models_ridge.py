from pathlib import Path
import sys
import pandas as pd
import numpy as np
import tempfile
import pytest
from sklearn.model_selection import train_test_split 
local_module_path = Path("../modules/model_wrapper")
sys.path.append(local_module_path.as_posix())
local_module_path = Path("../modules/data_transformers")
sys.path.append(local_module_path.as_posix())
local_module_path = Path("../modules/auxiliary")
sys.path.append(local_module_path.as_posix())
local_module_path = Path("../modules/constants")
sys.path.append(local_module_path.as_posix())

import data_transformers as dt
from model_wrapper import model_wrapper as mw
import constants as co

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from functools import partial

ARTIFACTS_DIR = Path(__file__).parent/"artifacts"

#@pytest.fixture
def test_SVM_Train_CV_regressor():
    
    test_data_fit = pd.read_csv((ARTIFACTS_DIR/"test_data_large.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data_fit["smiles"]
    target = test_data_fit["target"]
    
    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)

    pipeline_file = "test_pipeline_configuration_regression_pca_scaling.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    smiles_codes_train, smiles_codes_train_val, y_train, y_train_val = train_test_split(smiles_codes, y_train, test_size=0.1, random_state=co.RANDOM_STATE)
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes_train, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes_train)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    pca_columns = pipeline_train["PCA"].PCA_feature_names
    assert np.abs(X_train[pca_columns].sum().sum()) < co.EPSILON*1000.0
    assert np.abs(y_train.sum() - 5823.899535822553) < co.EPSILON
    assert np.abs(X_train["QED"].sum()) < co.EPSILON*100.0
    assert np.abs(X_train["SLogP"].sum()) < co.EPSILON*100.0
    assert np.abs(X_train["MW"].sum()) < co.EPSILON*100.0
    
    m_r = mw.M_Ridge(prediction_type=mw.Prediction_Type.REGRESSION, pipeline_configuration=pipeline_configuration)

    m_r.Initialize_Seeds(co.RANDOM_STATE)
    
    data = {"smiles_codes_train": smiles_codes_train, "targets_train": y_train, 
            "smiles_codes_val": smiles_codes_train_val, "targets_val": y_train_val}
    
    m_r.Data = data
    m_r.Create_Features()
    
    preds = m_r.Train_CV(hyperparameters={}, n_outer=1, n_cv=3)
    assert np.abs(preds.sum() - 5866.250526520913) < co.EPSILON
    assert np.abs(preds.mean() - 6.518056140578792) < co.EPSILON
    
    
def test_SVM_Train_CV_classifier():
    
    test_data_fit = pd.read_csv((ARTIFACTS_DIR/"test_data_large.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data_fit["smiles"]
    target = test_data_fit["target"]
    
    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)

    pipeline_file = "test_pipeline_configuration_classification_pca_scaling.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    smiles_codes_train, smiles_codes_train_val, y_train, y_train_val = train_test_split(smiles_codes, y_train, test_size=0.1, random_state=co.RANDOM_STATE)
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes_train, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes_train)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    pca_columns = pipeline_train["PCA"].PCA_feature_names
    assert np.abs(X_train[pca_columns].sum().sum()) < co.EPSILON*1000.0
    assert np.abs(y_train.sum() - 753) < co.EPSILON
    assert np.abs(X_train["QED"].sum()) < co.EPSILON*100.0
    assert np.abs(X_train["SLogP"].sum()) < co.EPSILON*100.0
    assert np.abs(X_train["MW"].sum()) < co.EPSILON*100.0
    
    m_c = mw.M_Ridge(prediction_type=mw.Prediction_Type.CLASSIFICATION, pipeline_configuration=pipeline_configuration)

    m_c.Initialize_Seeds(co.RANDOM_STATE)
    
    data = {"smiles_codes_train": smiles_codes_train, "targets_train": y_train, 
            "smiles_codes_val": smiles_codes_train_val, "targets_val": y_train_val}
    
    m_c.Data = data
    m_c.Create_Features()
    
    preds = m_c.Train_CV(hyperparameters={}, n_outer=1, n_cv=3)
    assert np.abs(preds.sum() - 769.0) < co.EPSILON
    assert np.abs(preds.mean() - 0.8544444444444445) < co.EPSILON