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

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from functools import partial

ARTIFACTS_DIR = Path(__file__).parent/"artifacts"

#@pytest.fixture
def test_MLP_Train_CV_regressor():
    
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
    assert np.abs(X_train[pca_columns].sum().sum()) < co.EPSILON*100.0
    assert np.abs(y_train.sum() - 381.77915051762307) < co.EPSILON
    assert np.abs(X_train["QED"].sum()) < co.EPSILON*100.0
    assert np.abs(X_train["SLogP"].sum()) < co.EPSILON*100.0
    assert np.abs(X_train["MW"].sum()) < co.EPSILON*100.0
    
    m_r = mw.M_MLP(prediction_type=mw.Prediction_Type.REGRESSION, pipeline_configuration=pipeline_configuration)
    m_r.Validate = True
    m_r.Initialize_Seeds(co.RANDOM_STATE)
    # for the repeatability concerns the calculation runs on CPU, with the GPU it is not repeatable
    m_r.Force_CPU()
    
    data = {"smiles_codes_train": smiles_codes_train, "targets_train": y_train, 
            "smiles_codes_val": smiles_codes_train_val, "targets_val": y_train_val}
    
    m_r.Data = data
    m_r.Create_Features()
    
    preds = m_r.Train_CV(hyperparameters={}, n_outer=1, n_cv=3)
    assert np.abs(preds.sum() - 456.1112973988056) < co.EPSILON
    assert np.abs(preds.mean() - 0.577356072656716) < co.EPSILON
    
    # create final model
    m_r.Train(X=X_train, y=y_train, hyperparameters={})
    some_smiles = "CC(=O)O[C@H]1C(=O)[C@]2(C)[C@@H](O)C[C@H]3OC[C@@]3(OC(C)=O)[C@H]2[C@H](OC(=O)c2ccccc2)[C@]2(O)C[C@H](OC(=O)[C@H](O)[C@@H](NC(=O)c3ccccc3)c3ccccc3)C(C)=C1C2(C)C"
    some_smiles = pd.Series(some_smiles)
    some_smile_features = dt.create_molecular_features(pipeline=pipeline_train, smiles_codes=some_smiles)
    some_smile_features = dt.process_molecular_features(pipeline=pipeline_train, X=some_smile_features)
    
    activity = m_r.Predict(some_smile_features)
    
    remove_me = "remove_me.model"
    m_r.Save(remove_me)
    
    m_r_2 = mw.M_MLP(prediction_type=mw.Prediction_Type.REGRESSION, pipeline_configuration=None)
    
    m_r_2.Load(remove_me)
    m_r_2.Validate = True
    m_r_2.Initialize_Seeds(co.RANDOM_STATE)
    # for the repeatability concerns the calculation runs on CPU, with the GPU it is not repeatable
    m_r_2.Force_CPU()
    activity_2 = m_r_2.Predict(some_smile_features)
    assert np.abs(activity[0] - activity_2[0]) < co.EPSILON
    
    
def test_MLP_Train_CV_classifier():
    
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
    assert np.abs(X_train[pca_columns].sum().sum()) < co.EPSILON*100.0
    assert np.abs(y_train.sum() - 753) < co.EPSILON
    assert np.abs(X_train["QED"].sum()) < co.EPSILON*100.0
    assert np.abs(X_train["SLogP"].sum()) < co.EPSILON*100.0
    assert np.abs(X_train["MW"].sum()) < co.EPSILON*100.0
    
    m_c = mw.M_MLP(prediction_type=mw.Prediction_Type.CLASSIFICATION, pipeline_configuration=pipeline_configuration)
    m_c.Validate = True
    m_c.Initialize_Seeds(co.RANDOM_STATE)
    # for the repeatability concerns the calculation runs on CPU, with the GPU it is not repeatable
    m_c.Force_CPU()
    
    data = {"smiles_codes_train": smiles_codes_train, "targets_train": y_train, 
            "smiles_codes_val": smiles_codes_train_val, "targets_val": y_train_val}
    
    m_c.Data = data
    m_c.Create_Features()
    
    preds = m_c.Train_CV(hyperparameters={}, n_outer=1, n_cv=3)
    assert np.abs(preds.sum() - 647.1320064365864) < co.EPSILON
    assert np.abs(preds.mean() - 0.7190355627073182) < co.EPSILON
    
    
def test_MLP_hyperopt_regressor():
    test_data_fit = pd.read_csv((ARTIFACTS_DIR/"test_data_large.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data_fit["smiles"]
    target = test_data_fit["target"]

    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)

    pipeline_file = "pipeline_configuration_MLP_regression.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    pipeline = dt.create_pipeline(pipeline_configuration=pipeline_configuration["feature_transform"])

    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]

    smiles_codes_train, smiles_codes_train_val, y_train, y_train_val = train_test_split(smiles_codes, y_train, test_size=0.1, random_state=co.RANDOM_STATE)

    m_r = mw.M_MLP(prediction_type=mw.Prediction_Type.REGRESSION, pipeline_configuration=pipeline_configuration)
    data = {"smiles_codes_train": smiles_codes_train, "targets_train": y_train, 
            "smiles_codes_val": smiles_codes_train_val, "targets_val": y_train_val}
    m_r.Data = data
    m_r.Validate = True
    m_r.Initialize_Seeds(co.RANDOM_STATE)
    m_r.Force_CPU()
    m_r.Create_Features()

    aux_data = {"run_name": "test", "n_outer": 1, "n_cv": 4, "development": False, "goal_function": "mse", "goal_function_multiplier": 1.0, "threshold": None,
                "track_model": False, "experiment": None, "comment": None, "max_evals": None}

    fmin_objective = partial(m_r.F_Opt, aux_data=aux_data)
    trials = Trials()
    rstate = np.random.RandomState(co.RANDOM_STATE+1)
    max_evals = 2
    best_hyperparams = fmin(fn=fmin_objective, space=m_r.Space, algo=tpe.suggest, max_evals=max_evals, 
                            trials=trials, return_argmin=True, rstate=rstate, show_progressbar=True)
    
    best_hyperparams = space_eval(space=m_r.Space, hp_assignment=best_hyperparams)
    measure = m_r.Get_Measure(smiles_codes, y_train, hyperparameters=best_hyperparams, n_outer=aux_data["n_outer"], n_cv=aux_data["n_cv"],
                        threshold=None, goal=aux_data["goal_function"]) 
    assert np.abs(measure - 0.8542637369202594) < co.EPSILON
    
def test_MLP_hyperopt_classifier():
    test_data_fit = pd.read_csv((ARTIFACTS_DIR/"test_data_large.csv").absolute().as_posix(), sep=",", index_col="molregno")
    smiles_codes = test_data_fit["smiles"]
    target = test_data_fit["target"]

    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)

    pipeline_file = "pipeline_configuration_MLP_classification.json"
    pipeline_configuration = dt.read_in_pipeline(pipeline_file=pipeline_file, pipeline_directory=ARTIFACTS_DIR)
    pipeline = dt.create_pipeline(pipeline_configuration=pipeline_configuration["feature_transform"])

    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]

    smiles_codes_train, smiles_codes_train_val, y_train, y_train_val = train_test_split(smiles_codes, y_train, test_size=0.1, random_state=co.RANDOM_STATE)

    m_r = mw.M_MLP(prediction_type=mw.Prediction_Type.CLASSIFICATION, pipeline_configuration=pipeline_configuration)
    data = {"smiles_codes_train": smiles_codes_train, "targets_train": y_train, 
            "smiles_codes_val": smiles_codes_train_val, "targets_val": y_train_val}
    m_r.Data = data
    m_r.Validate = True
    m_r.Initialize_Seeds(co.RANDOM_STATE)
    m_r.Force_CPU()
    m_r.Create_Features()

    aux_data = {"run_name": "test", "n_outer": 1, "n_cv": 4, "development": False, "goal_function": "average_precision", "goal_function_multiplier": 1.0, "threshold": None,
                "track_model": False, "experiment": None, "comment": None, "max_evals": None}

    fmin_objective = partial(m_r.F_Opt, aux_data=aux_data)
    trials = Trials()
    rstate = np.random.RandomState(co.RANDOM_STATE+1)
    max_evals = 2
    best_hyperparams = fmin(fn=fmin_objective, space=m_r.Space, algo=tpe.suggest, max_evals=max_evals, 
                            trials=trials, return_argmin=True, rstate=rstate, show_progressbar=True)
    
    best_hyperparams = space_eval(space=m_r.Space, hp_assignment=best_hyperparams)
    measure = m_r.Get_Measure(smiles_codes, y_train, hyperparameters=best_hyperparams, n_outer=aux_data["n_outer"], n_cv=aux_data["n_cv"],
                        threshold=None, goal=aux_data["goal_function"]) 
    assert np.abs(measure - 0.6334213390183097) < co.EPSILON