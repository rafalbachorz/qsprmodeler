
from sklearn.model_selection import train_test_split
import data_transformers as dt
from model_wrapper import model_wrapper as mw
import constants as co
from hyperopt import Trials, fmin, tpe, space_eval
from hyperopt.fmin import generate_trials_to_calculate
import pandas as pd
import numpy as np
from functools import partial
from auxiliary import loggers
l = loggers.get_logger(logger_name="logger")

from enum import Enum
# class ColumnNames(Enum):
#     SMILES_COLUMN = "canonical_smiles_get_levels"
#     TARGET_COLUMN = "IC50_nM"

def train(train_data: pd.DataFrame, configuration_train: dict, configuration_data:dict, configuration_storage: dict, prediction_methodology: str, 
          prediction_type: str, GPU_support: bool):
    """_summary_

    Args:
        train_data (pd.DataFrame): the train data
        configuration_train (dict): configuration of the training
        configuration_storage (dict): configuration of the model storage

    Returns:
        _type_: _description_
    """
    smiles_column = configuration_data["molecule_column"]
    target_column = configuration_data["target_column"]
    smiles_codes = train_data[smiles_column].copy()
    target = train_data[target_column].copy()

    non_idx = dt.validate_smiles(smiles_codes=smiles_codes)
    smiles_codes = smiles_codes.drop(non_idx, axis=0)
    target = target.drop(non_idx, axis=0)

    pipeline_configuration = dt.read_in_pipeline(pipeline_file=configuration_train["pipeline_file"], 
                                                 pipeline_directory=co.PIPELINE_DIR)

    y_train = dt.process_taget(pipeline_configuration=pipeline_configuration["target_transform"], y=target)
    smiles_codes = smiles_codes.loc[y_train.index]

    smiles_codes_train, smiles_codes_train_val, y_train, y_train_val = train_test_split(smiles_codes, y_train, test_size=0.1, random_state=co.RANDOM_STATE)
    
    if prediction_methodology == mw.Prediction_Methodology.MLP:
        model_wrapper = mw.M_MLP(prediction_type=prediction_type, pipeline_configuration=pipeline_configuration)
        model_wrapper.Validate = True
        model_wrapper.Verbose = 0
        #model_wrapper.Force_CPU()
    elif prediction_methodology == mw.Prediction_Methodology.XGBOOST:
        model_wrapper = mw.M_XGBoost(prediction_type=prediction_type, pipeline_configuration=pipeline_configuration)
        model_wrapper.Validate = True
        if GPU_support: model_wrapper.TreeMethod = "gpu_hist"
    elif prediction_methodology == mw.Prediction_Methodology.RF:
        model_wrapper = mw.M_RandomForest(prediction_type=prediction_type, pipeline_configuration=pipeline_configuration)
    elif prediction_methodology == mw.Prediction_Methodology.SVM:
        model_wrapper = mw.M_SVM(prediction_type=prediction_type, pipeline_configuration=pipeline_configuration)
        model_wrapper.Validate = False
    elif prediction_methodology == mw.Prediction_Methodology.RIDGE:
        model_wrapper = mw.M_Ridge(prediction_type=prediction_type, pipeline_configuration=pipeline_configuration)
    elif prediction_methodology == mw.Prediction_Methodology.BAGGING:
        model_wrapper = mw.M_Bagging(prediction_type=prediction_type, pipeline_configuration=pipeline_configuration)
    else:
        raise NotImplementedError
    model_wrapper.Initialize_Seeds(co.RANDOM_STATE)
    
    model_wrapper.Data = {"smiles_codes_train": smiles_codes_train, "targets_train": y_train, 
            "smiles_codes_val": smiles_codes_train_val, "targets_val": y_train_val}
    model_wrapper.Create_Features()
    
    
    
    fmin_objective = partial(model_wrapper.F_Opt, aux_data=configuration_train)
    trials = Trials()
    #default_hyperparameters = { key: model_wrapper.DefaultHyperparameters[key] for key in model_wrapper.HyperoptSpace.keys() }
    #default_hyperparameters = space_eval(model_wrapper.Space, default_hyperparameters)
    #trials = generate_trials_to_calculate([default_hyperparameters])
    # rstate = np.random.RandomState(co.RANDOM_STATE+1)
    rstate = np.random.default_rng(co.RANDOM_STATE)
    max_evals = configuration_train["max_evals"]
    best_hyperparams = fmin(fn=fmin_objective, space=model_wrapper.Space, algo=tpe.suggest, max_evals=max_evals, 
                            trials=trials, return_argmin=True, rstate=rstate, show_progressbar=True)
    optimization_history = {
        "best_trial": trials.best_trial,
        "results": trials.results,
        "vals": trials.vals
    }
    model_wrapper.OptimizationHistory = optimization_history
    model_wrapper.TrainingParams = configuration_train
    
    best_hyperparams = space_eval(space=model_wrapper.Space, hp_assignment=best_hyperparams)
    measures = model_wrapper.Get_All_Measures(smiles_codes, y_train, hyperparameters=best_hyperparams, 
                                   n_outer=configuration_train["n_outer"], n_cv=configuration_train["n_cv"],
                                   threshold=configuration_train["threshold"])
    model_wrapper.Measures = measures
    l.info("Regressor quality after hyperparameter optimization: "+str(measures))
    default_measures = model_wrapper.Get_All_Measures(smiles_codes, y_train, hyperparameters=best_hyperparams, 
                                   n_outer=configuration_train["n_outer"], n_cv=configuration_train["n_cv"],
                                   threshold=configuration_train["threshold"], default_hyperparameters=True)    
    model_wrapper.DefaultMeasures = default_measures
    l.info("Regressor quality after hyperparameter optimization: "+str(default_measures))
    # final training for model store
    model_wrapper.Pipeline.fit(X=smiles_codes_train, y=None)
    X_train = dt.create_molecular_features(pipeline=model_wrapper.Pipeline, smiles_codes=smiles_codes_train)
    X_train = dt.process_molecular_features(pipeline=model_wrapper.Pipeline, X=X_train)
    model_wrapper.History = model_wrapper.Train(X=X_train, y=y_train, hyperparameters=best_hyperparams)
    if prediction_methodology == mw.Prediction_Methodology.MLP:
        model_wrapper.History = model_wrapper.History.history.copy()
        
    model_wrapper.ModelStatus = mw.Model_Status.PRODUCTION
    
    m_r_file = configuration_storage["resulting_model"]
    model_wrapper.Save((co.MODEL_DIR/m_r_file).absolute().as_posix())
    
    return best_hyperparams, measures