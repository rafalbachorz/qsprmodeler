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

def test_XGBoost_Train_CV_with_scaling():
    
    test_data_fit = pd.read_csv((ARTIFACTS_DIR/"test_data_1.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data_fit["smiles"]
    target = test_data_fit["target"]
    
    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)

    pipeline_file = "test_pipeline_configuration_regression_scaling.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    smiles_codes_train, smiles_codes_train_val, y_train, y_train_val = train_test_split(smiles_codes, y_train, test_size=0.1, random_state=co.RANDOM_STATE)
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes_train, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes_train)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    fp_columns = pipeline_train["Molecular features"].fp_column_names
    assert X_train[fp_columns].sum().sum() == 3727
    assert np.abs(y_train.sum() - 296.8966552477734) < co.EPSILON
    assert np.abs(X_train["QED"].sum()) < co.EPSILON*100.0
    assert np.abs(X_train["SLogP"].sum()) < co.EPSILON*100.0
    assert np.abs(X_train["MW"].sum()) < co.EPSILON*100.0
    
    m_r = mw.M_XGBoost(prediction_type=mw.Prediction_Type.REGRESSION, pipeline_configuration=pipeline_configuration)

    data = {"smiles_codes_train": smiles_codes_train, "targets_train": y_train, 
            "smiles_codes_val": smiles_codes_train_val, "targets_val": y_train_val}
    
    m_r.Data = data
    m_r.Create_Features()
    
    preds = m_r.Train_CV(hyperparameters={}, n_outer=1, n_cv=3)
    assert np.abs(preds.mean() - 3.5262140231973986) < co.EPSILON


def test_XGBoost_Train_with_scaling():
    
    test_data_fit = pd.read_csv((ARTIFACTS_DIR/"test_data_1.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data_fit["smiles"]
    target = test_data_fit["target"]
    
    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)

    pipeline_file = "test_pipeline_configuration_regression_scaling.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    smiles_codes_train, smiles_codes_train_val, y_train, y_train_val = train_test_split(smiles_codes, y_train, test_size=0.1, random_state=co.RANDOM_STATE)
    
    pipeline_train = dt.create_pipeline(pipeline_configuration["feature_transform"])
    pipeline_train.fit(X=smiles_codes_train, y=None)
    X_train = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes_train)
    X_train = dt.process_molecular_features(pipeline=pipeline_train, X=X_train)
    
    X_train_val = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=smiles_codes_train_val)
    X_train_val = dt.process_molecular_features(pipeline=pipeline_train, X=X_train_val)
    
    m_r = mw.M_XGBoost(prediction_type=mw.Prediction_Type.REGRESSION, pipeline_configuration=pipeline_configuration)

    data = {"smiles_codes_train": smiles_codes, "targets_train": y_train, 
            "smiles_codes_val": smiles_codes_train_val, "targets_val": y_train_val}
    m_r.Data = data
    m_r.Create_Features()
    
    m_r.Train(X_train, y_train, hyperparameters={})
    preds = m_r.Predict(X_train_val)
    assert np.abs(preds.mean() - 3.5546412467956543) < co.EPSILON

def test_XGBoost_Get_Measure_with_scaling():
    
    test_data_fit = pd.read_csv((ARTIFACTS_DIR/"test_data_1.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data_fit["smiles"]
    target = test_data_fit["target"]
    
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
    assert X_train[fp_columns].sum().sum() == 4123
    assert np.abs(y_train.sum() - 332.9964572007816) < co.EPSILON
    assert np.abs(X_train["QED"].sum()) < co.EPSILON*100.0
    assert np.abs(X_train["SLogP"].sum()) < co.EPSILON*100.0
    assert np.abs(X_train["MW"].sum()) < co.EPSILON*100.0
    
    m_r = mw.M_XGBoost(prediction_type=mw.Prediction_Type.REGRESSION, pipeline_configuration=pipeline_configuration)

    data = {"smiles_codes_train": smiles_codes, "targets_train": y_train, 
            "smiles_codes_val": None, "targets_val": None}
    m_r.Data = data
    m_r.Create_Features()
    
    aux_data = {"run_name": "test",
            "n_outer": 1, "n_cv": 4, "development": False,
            "goal_function": "mse", "goal_function_multiplier": 1.0, "threshold": None,
            "track_model": False, "experiment": None, "comment": None, "max_evals": None}
    
    
    mse = m_r.Get_Measure(smiles_codes, y_train, hyperparameters={}, n_outer=aux_data["n_outer"], n_cv=aux_data["n_cv"],
                    threshold=None, goal=aux_data["goal_function"])
    assert np.abs(mse - 0.8757378574512518) < co.EPSILON
  
  
def test_XGBoost_Get_Measure_with_scaling_classification():
    
    test_data_fit = pd.read_csv((ARTIFACTS_DIR/"test_data_1.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data_fit["smiles"]
    target = test_data_fit["target"]
    
    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)

    pipeline_file = "test_pipeline_configuration_classification_no_o_r.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]
    
    m_c = mw.M_XGBoost(prediction_type=mw.Prediction_Type.CLASSIFICATION, pipeline_configuration=pipeline_configuration)

    data = {"smiles_codes_train": smiles_codes, "targets_train": y_train, 
            "smiles_codes_val": None, "targets_val": None}
    m_c.Data = data
    m_c.Create_Features()
    
    aux_data = {"run_name": "test",
            "n_outer": 1, "n_cv": 4, "development": False,
            "goal_function": "average_precision", "goal_function_multiplier": -1.0, "threshold": 0.5,
            "track_model": False, "experiment": None, "comment": None, "max_evals": None}
    
    measure = m_c.Get_Measure(smiles_codes, y_train, hyperparameters={}, n_outer=aux_data["n_outer"], n_cv=aux_data["n_cv"],
                    threshold=aux_data["threshold"], goal=aux_data["goal_function"])
    assert np.abs(measure - 0.020718816067653276) < co.EPSILON  
  
def test_XGBoost_hyperopt():

    test_data_fit = pd.read_csv((ARTIFACTS_DIR/"test_data_1.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data_fit["smiles"]
    target = test_data_fit["target"]

    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)

    pipeline_file = "pipeline_configuration_XGBoost_regression.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=co.PIPELINE_DIR)
    
    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]

    m_r = mw.M_XGBoost(prediction_type=mw.Prediction_Type.REGRESSION, pipeline_configuration=pipeline_configuration)
    
    data = {"smiles_codes_train": smiles_codes, "targets_train": y_train, 
            "smiles_codes_val": None, "targets_val": None}
    m_r.Data = data
    m_r.Create_Features()

    aux_data = {"run_name": "test",
            "n_outer": 1, "n_cv": 4, "development": False,
            "goal_function": "mse", "goal_function_multiplier": 1.0, "threshold": None,
            "track_model": False, "experiment": None, "comment": None, "max_evals": None}

    fmin_objective = partial(m_r.F_Opt, aux_data=aux_data)
    trials = Trials()
    rstate = np.random.RandomState(co.RANDOM_STATE+1)
    max_evals = 5
    best_hyperparams = fmin(fn=fmin_objective, space=m_r.Space, algo=tpe.suggest, max_evals=max_evals, 
                            trials=trials, return_argmin=True, rstate=rstate)
    
    m_r.OptimizationHistory = trials
    
    mse = m_r.Get_Measure(smiles_codes, y_train, hyperparameters=best_hyperparams, n_outer=aux_data["n_outer"], n_cv=aux_data["n_cv"],
                          threshold=None, goal=aux_data["goal_function"])
    
    assert np.abs(mse - 0.9372528945602626) < co.EPSILON
    
    tmp_file = tempfile.NamedTemporaryFile()
    m_r.Save(tmp_file.name)
    m_r_2 = mw.M_XGBoost(prediction_type=mw.Prediction_Type.REGRESSION, pipeline_configuration=None)
    
    m_r_2.Load(tmp_file.name)
    mse_2 = m_r.Get_Measure(smiles_codes, y_train, hyperparameters=best_hyperparams, n_outer=aux_data["n_outer"], n_cv=aux_data["n_cv"],
                            threshold=None, goal=aux_data["goal_function"])
    tmp_file.close()
    assert np.abs(mse - mse_2) < co.EPSILON