import argparse
import logging
import json
from turtle import st
import pandas as pd
import numpy as np

from pathlib import Path

import data_transformers as dt
from model_wrapper import model_wrapper as mw
from model_wrapper import train
import constants as co
from auxiliary import loggers
from enum import Enum
from functools import partial
from sklearn.model_selection import train_test_split
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from hyperopt.fmin import generate_trials_to_calculate

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

l = loggers.get_logger(logger_name="logger")

class ColumnNames(Enum):
    SMILES_COLUMN = "canonical_smiles_get_levels"
    TARGET_COLUMN = "IC50_nM"

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-tc", "--training_conf", type=str, help="Training configuration", required=True)
    args = parser.parse_args()
    
    l.info("Training configuration: "+args.training_conf)
    with open(co.TRAINING_CONF_DIR/args.training_conf, "r") as f:
        training_params = json.load(f)

    l.info("Preparing the raw data sets...")    
    train_data = dt.data_load(training_params["data_preparation"])

    l.info("Training the model...")
    best_hyperparams, measure = train(train_data=train_data, configuration_train=training_params["training_aux_data"],
                                      configuration_data=training_params["data_preparation"],
                                      configuration_storage=training_params["model_storage"],
                                      prediction_methodology=training_params["prediction_methodology"],
                                      prediction_type=training_params["prediction_type"],
                                      GPU_support=training_params["GPU_support"])
    l.info("Finishing the calculations...")
    

    
        
    
        


   
    
    
    
    

