from pathlib import Path

root = Path(__file__).parent.parent.parent

RANDOM_STATE = 42
N_JOBS = 8
ERROR_GOAL_FUCTION = 999.0
EPSILON = 1e-14
PIPELINE_DIR = root/"pipelines"
DATA_DIR = root/"data"/"processed"
DATA_DIR_CANDIDATES = root/"data"/"candidates"
MODEL_DIR = root/"models"
TRAINING_CONF_DIR = root/"training_configurations"