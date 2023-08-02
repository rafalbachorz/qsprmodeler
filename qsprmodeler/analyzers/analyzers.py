from auxiliary import plotScatterWithIdeal, create_precision_recall_plot, create_roc_plot
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, precision_score, r2_score, precision_recall_curve, roc_auc_score, average_precision_score, precision_score, recall_score, \
ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, f1_score, accuracy_score, matthews_corrcoef, mean_squared_log_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd
import tensorflow as tf
import random as rn
import json
from pathlib import Path
import  constants as co
from model_wrapper import model_wrapper as mw
from data_transformers import create_molecular_features, process_molecular_features, read_in_pipeline
from auxiliary import loggers

l = loggers.get_logger(logger_name="logger")
SEP = ","

def summarize_calculation(training_configuration_file: str):
    with open((training_configuration_file).as_posix(), "r") as f:
        training_params = json.load(f)
    with open(training_params["training_aux_data"]["pipeline_file"], "r") as f:
        pipeline = json.load(f)
    l.info("Training configuration:")
    l.info("Prediction methodology: %s", training_params["prediction_methodology"])
    l.info("Prediction type: %s", training_params["prediction_type"])
    l.info("Data file: %s", training_params["data_preparation"]["data_file"])
    l.info("Max. activity level: %d", training_params["data_preparation"]["max_level_activity"])
    l.info("Std. threshold: %d", training_params["data_preparation"]["std_threshold"])
    l.info("Data preparation strategy: %s", training_params["data_preparation"]["strategy"])
    l.info("Model storage: %s", training_params["model_storage"]["resulting_model"])
    l.info("Pipeline configuration:")
    calculate_fps = pipeline["feature_transform"]["Calculate_molecular_features"]["calculate_fps"]
    l.info("FPs?: %s", calculate_fps)
    if calculate_fps=="yes":
        l.info("FP size?: %s", pipeline["feature_transform"]["Calculate_molecular_features"]["fingerprint_size"])
    l.info("MDs?: %s", pipeline["feature_transform"]["Calculate_molecular_features"]["calculate_descriptors"])
    l.info("Scaling?: %s", pipeline["feature_transform"]["Scaling"]["proceed"])
    proceed_pca = pipeline["feature_transform"]["PCA"]["proceed"]
    l.info("PCA?: %s", proceed_pca)
    if proceed_pca=="yes":
        l.info("PCA - apply to MD, FP?: %s, %s", pipeline["feature_transform"]["PCA"]["apply_to_md"], pipeline["feature_transform"]["PCA"]["apply_to_fp"])
        l.info("PCA - number of components: %d", pipeline["feature_transform"]["PCA"]["n_components"])
    l.info("Target transform?: %s", pipeline["target_transform"]["Target_transform"]["proceed"])
    l.info("Target binarization?: %s", pipeline["target_transform"]["Target_binarization"]["proceed"])

def create_model_label(training_params: dict) -> str:
    pipeline = read_in_pipeline(pipeline_file=training_params["training_aux_data"]["pipeline_file"])
    label = ""
    if pipeline["feature_transform"]["Calculate_molecular_features"]["proceed"] == "yes":
        if pipeline["feature_transform"]["Calculate_molecular_features"]["calculate_fps"] == "yes": label += "FP_"
        if pipeline["feature_transform"]["Calculate_molecular_features"]["calculate_descriptors"] == "yes": label += "MD_"
    if pipeline["feature_transform"]["Scaling"]["proceed"] == "yes":
        label += "Scaling_"
    else:
        label += "NoScaling_"
    
    if pipeline["feature_transform"]["PCA"]["proceed"] == "yes":
        label += "PCA_"+str(pipeline["feature_transform"]["PCA"]["n_components"])+"_"
    else:
        label += "NoPCA_"
    label += training_params["data_preparation"]["strategy"]+"_"+str(training_params["data_preparation"]["std_threshold"])
    return label

def get_predictions_classifier(training_configuration_file):
    
    with open(training_configuration_file.as_posix(), "r") as f:
        training_params = json.load(f)

    prediction_methodology = training_params["prediction_methodology"]
    prediction_type=training_params["prediction_type"]

    if prediction_methodology == mw.Prediction_Methodology.MLP:
        m_c = mw.M_MLP(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.XGBOOST:
        m_c = mw.M_XGBoost(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.RF:
        m_c = mw.M_RandomForest(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.SVM:
        m_c = mw.M_SVM(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.RIDGE:
        m_c = mw.M_Ridge(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.BAGGING:
        m_c = mw.M_Bagging(prediction_type=prediction_type, pipeline_configuration=None)
    else:
        raise NotImplementedError

    model_label = prediction_methodology + "_" + create_model_label(training_params)

    model_file = training_params["model_storage"]["resulting_model"]
    m_c.Load(model_file)
    l.info("Optimal hyperparameters:, %s", str(m_c.Hyperparameters))

    l.info("Performance on external (hold-out) set")
    smiles_codes_val = m_c.Data["smiles_codes_val"]
    smiles_codes_train = m_c.Data["smiles_codes_train"]
    m_c.Pipeline.fit(X=smiles_codes_train, y=None)
    X_val = create_molecular_features(pipeline=m_c.Pipeline, smiles_codes=smiles_codes_val)
    X_val = process_molecular_features(pipeline=m_c.Pipeline, X=X_val)
    y_train_val_pred = m_c.Predict(X_val)
    y_train_val = m_c.Data["targets_val"]
    
    l.info("Performance on training set")
    smiles_codes_train = m_c.Data["smiles_codes_train"]
    m_c.Pipeline.fit(X=smiles_codes_train, y=None)
    X_train = create_molecular_features(pipeline=m_c.Pipeline, smiles_codes=smiles_codes_train)
    X_train = process_molecular_features(pipeline=m_c.Pipeline, X=X_train)
    y_train_pred = m_c.Predict(X_train)
    y_train = m_c.Data["targets_train"]

    return smiles_codes_train, y_train, y_train_pred, smiles_codes_val, y_train_val, y_train_val_pred

def analyze_classifier(training_configuration_file, show_tsne=False, result_file=None):
    summarize_calculation(training_configuration_file)
    with open(training_configuration_file.as_posix(), "r") as f:
        training_params = json.load(f)

    if result_file:    
        rf = Path(result_file)
        fle = open(rf.as_posix(), "a")

    prediction_methodology = training_params["prediction_methodology"]
    prediction_type=training_params["prediction_type"]

    if prediction_methodology == mw.Prediction_Methodology.MLP:
        m_c = mw.M_MLP(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.XGBOOST:
        m_c = mw.M_XGBoost(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.RF:
        m_c = mw.M_RandomForest(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.SVM:
        m_c = mw.M_SVM(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.RIDGE:
        m_c = mw.M_Ridge(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.BAGGING:
        m_c = mw.M_Bagging(prediction_type=prediction_type, pipeline_configuration=None)
    else:
        raise NotImplementedError

    model_label = prediction_methodology + "_" + create_model_label(training_params)

    #m_c = mw.M_XGBoost(prediction_type=mw.Prediction_Type.CLASSIFICATION, pipeline_configuration=None)
    model_file = training_params["model_storage"]["resulting_model"]
    m_c.Load(model_file)
    l.info("Optimal hyperparameters:, %s", str(m_c.Hyperparameters))
    l.info("Performance on external (hold-out) set")
    smiles_codes_val = m_c.Data["smiles_codes_val"]
    smiles_codes_train = m_c.Data["smiles_codes_train"]
    m_c.Pipeline.fit(X=smiles_codes_train, y=None)
    X_val = create_molecular_features(pipeline=m_c.Pipeline, smiles_codes=smiles_codes_val)
    X_val = process_molecular_features(pipeline=m_c.Pipeline, X=X_val)
    y_train_val_pred = m_c.Predict(X_val)
    y_train_val = m_c.Data["targets_val"]


    display = ConfusionMatrixDisplay.from_predictions(y_train_val, y_train_val_pred > training_params["training_aux_data"]["threshold"])
    _ = display.ax_.set_title("CM: "+model_label+"\nTest set")
    display.figure_.savefig(model_label+"_cm_test.pdf", dpi=300, format="pdf")
    
    pr = precision_score(y_train_val, y_train_val_pred > training_params["training_aux_data"]["threshold"])
    re = recall_score(y_train_val, y_train_val_pred > training_params["training_aux_data"]["threshold"])
    acc = accuracy_score(y_train_val, y_train_val_pred > training_params["training_aux_data"]["threshold"])
    f1 = f1_score(y_train_val, y_train_val_pred > training_params["training_aux_data"]["threshold"])
    rocauc = roc_auc_score(y_train_val, y_train_val_pred > training_params["training_aux_data"]["threshold"])
    ap = average_precision_score(y_train_val, y_train_val_pred > training_params["training_aux_data"]["threshold"])
    mcc = matthews_corrcoef(y_train_val, y_train_val_pred > training_params["training_aux_data"]["threshold"])
    l.info("Precision on hold-out: %f", pr)
    l.info("Recall on hold-out: %f", re)
    l.info("Accuracy on hold-out: %f", acc)
    l.info("F1-score on hold-out: %f", f1)
    l.info("ROC-AUC on hold-out: %f", rocauc)
    l.info("Average precision on hold-out: %f", ap)
    l.info("MCC on hold-out: %f", mcc)
    
    entry_line = model_label+"_test"+SEP+str(pr)+SEP+str(re)+SEP+str(acc)+SEP+str(f1)+SEP+str(rocauc)+SEP+str(ap)+SEP+str(mcc)+"\n"
    if result_file: fle.write(entry_line)
    
    display = PrecisionRecallDisplay.from_predictions(y_train_val, y_train_val_pred, name=prediction_methodology+" classifier")
    _ = display.ax_.set_title("PR plot: "+model_label+"\nTest set")
    _ = display.ax_.grid()
    display.figure_.savefig(model_label+"_pr_test.pdf", dpi=300, format="pdf")
    
    display = RocCurveDisplay.from_predictions(y_train_val, y_train_val_pred, name=prediction_methodology+" classifier")
    _ = display.ax_.set_title("ROC plot: "+model_label+"\nTest set")
    _ = display.ax_.grid()
    display.figure_.savefig(model_label+"_roc_test.pdf", dpi=300, format="pdf")

    l.info("Performance on training set")
    smiles_codes_train = m_c.Data["smiles_codes_train"]
    m_c.Pipeline.fit(X=smiles_codes_train, y=None)
    X_train = create_molecular_features(pipeline=m_c.Pipeline, smiles_codes=smiles_codes_train)
    X_train = process_molecular_features(pipeline=m_c.Pipeline, X=X_train)
    y_train_pred = m_c.Predict(X_train)
    y_train = m_c.Data["targets_train"]
    display = ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred > training_params["training_aux_data"]["threshold"])
    _ = display.ax_.set_title("CM: "+model_label+"\nTraining set")
    display.figure_.savefig(model_label+"_cm_training.pdf", dpi=300, format="pdf")
    
    pr = precision_score(y_train, y_train_pred > training_params["training_aux_data"]["threshold"])
    re = recall_score(y_train, y_train_pred > training_params["training_aux_data"]["threshold"])
    acc = accuracy_score(y_train, y_train_pred > training_params["training_aux_data"]["threshold"])
    f1 = f1_score(y_train, y_train_pred > training_params["training_aux_data"]["threshold"])
    rocauc = roc_auc_score(y_train, y_train_pred > training_params["training_aux_data"]["threshold"])
    ap = average_precision_score(y_train, y_train_pred > training_params["training_aux_data"]["threshold"])
    mcc = matthews_corrcoef(y_train, y_train_pred > training_params["training_aux_data"]["threshold"])
    
    l.info("Precision on training: %f", pr)
    l.info("Recall on training: %f", re)
    l.info("Accuracy on training: %f", acc)
    l.info("F1-score on training: %f", f1)
    l.info("ROC-AUC on training: %f", rocauc)
    l.info("Average precision on training: %f", ap)
    l.info("MCC on training %f", mcc)

    entry_line = model_label+"_training"+SEP+str(pr)+SEP+str(re)+SEP+str(acc)+SEP+str(f1)+SEP+str(rocauc)+SEP+str(ap)+SEP+str(mcc)+"\n"
    if result_file: fle.write(entry_line)

    display = PrecisionRecallDisplay.from_predictions(y_train, y_train_pred, name=prediction_methodology+" classifier")
    _ = display.ax_.set_title("PR plot: "+model_label+"\nTraining set")
    _ = display.ax_.grid()
    display.figure_.savefig(model_label+"_pr_training.pdf", dpi=300, format="pdf")
    
    display = RocCurveDisplay.from_predictions(y_train, y_train_pred, name=prediction_methodology+" classifier")
    _ = display.ax_.set_title("ROC plot: "+model_label+"\nTraining set")
    _ = display.ax_.grid()
    display.figure_.savefig(model_label+"_roc_training.pdf", dpi=300, format="pdf")
    
    if result_file: fle.close()


def get_predictions_regressor(training_configuration_file):
    with open(training_configuration_file.as_posix(), "r") as f:
        training_params = json.load(f)
    
    m_r = mw.M_XGBoost(prediction_type=mw.Prediction_Type.REGRESSION, pipeline_configuration=None)
    model_file = training_params["model_storage"]["resulting_model"]
    m_r.Load(model_file)

    prediction_methodology = training_params["prediction_methodology"]
    prediction_type=training_params["prediction_type"]

    if prediction_methodology == mw.Prediction_Methodology.MLP:
        m_r = mw.M_MLP(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.XGBOOST:
        m_r = mw.M_XGBoost(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.RF:
        m_r = mw.M_RandomForest(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.SVM:
        m_r = mw.M_SVM(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.RIDGE:
        m_r = mw.M_Ridge(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.BAGGING:
        m_r = mw.M_Bagging(prediction_type=prediction_type, pipeline_configuration=None)
    else:
        raise NotImplementedError

    model_label = prediction_methodology + "_" + create_model_label(training_params)

    model_file = training_params["model_storage"]["resulting_model"]
    m_r.Load(model_file)
    l.info("Optimal hyperparameters:, %s", str(m_r.Hyperparameters))
    smiles_codes_val = m_r.Data["smiles_codes_val"]
    smiles_codes_train = m_r.Data["smiles_codes_train"]
    m_r.Pipeline.fit(X=smiles_codes_train, y=None)
    X_val = create_molecular_features(pipeline=m_r.Pipeline, smiles_codes=smiles_codes_val)
    X_val = process_molecular_features(pipeline=m_r.Pipeline, X=X_val)
    y_train_val_pred = m_r.Predict(X_val)
    y_train_val = m_r.Data["targets_val"]


    smiles_codes_val = m_r.Data["smiles_codes_val"]
    smiles_codes_train = m_r.Data["smiles_codes_train"]
    m_r.Pipeline.fit(X=smiles_codes_train, y=None)
    X_train = create_molecular_features(pipeline=m_r.Pipeline, smiles_codes=smiles_codes_train)
    X_train = process_molecular_features(pipeline=m_r.Pipeline, X=X_train)
    y_train_pred = m_r.Predict(X_train)
    y_train = m_r.Data["targets_train"]
    
    return smiles_codes_train, y_train, y_train_pred, smiles_codes_val, y_train_val, y_train_val_pred

        
def analyze_regressor(training_configuration_file, result_file=None):
    summarize_calculation(training_configuration_file)
    with open(training_configuration_file.as_posix(), "r") as f:
        training_params = json.load(f)
    
    if result_file:    
        rf = Path(result_file)
        fle = open(rf.as_posix(), "a")
    
    m_r = mw.M_XGBoost(prediction_type=mw.Prediction_Type.REGRESSION, pipeline_configuration=None)
    model_file = training_params["model_storage"]["resulting_model"]
    m_r.Load(model_file)

    prediction_methodology = training_params["prediction_methodology"]
    prediction_type=training_params["prediction_type"]

    if prediction_methodology == mw.Prediction_Methodology.MLP:
        m_r = mw.M_MLP(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.XGBOOST:
        m_r = mw.M_XGBoost(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.RF:
        m_r = mw.M_RandomForest(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.SVM:
        m_r = mw.M_SVM(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.RIDGE:
        m_r = mw.M_Ridge(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.BAGGING:
        m_r = mw.M_Bagging(prediction_type=prediction_type, pipeline_configuration=None)
    else:
        raise NotImplementedError

    model_label = prediction_methodology + "_" + create_model_label(training_params)

    model_file = training_params["model_storage"]["resulting_model"]
    m_r.Load(model_file)
    l.info("Optimal hyperparameters:, %s", str(m_r.Hyperparameters))
    smiles_codes_val = m_r.Data["smiles_codes_val"]
    smiles_codes_train = m_r.Data["smiles_codes_train"]
    m_r.Pipeline.fit(X=smiles_codes_train, y=None)
    X_val = create_molecular_features(pipeline=m_r.Pipeline, smiles_codes=smiles_codes_val)
    X_val = process_molecular_features(pipeline=m_r.Pipeline, X=X_val)
    y_train_val_pred = m_r.Predict(X_val)
    y_train_val = m_r.Data["targets_val"]

    
    r2s = r2_score(y_train_val, y_train_val_pred)
    mse = mean_squared_error(y_train_val, y_train_val_pred)
    msle = mean_squared_log_error(y_train_val, y_train_val_pred)
    mae = mean_absolute_error(y_train_val, y_train_val_pred)
    mape = mean_absolute_percentage_error(y_train_val, y_train_val_pred)
    
    _, fig = plotScatterWithIdeal(y_train_val.values, y_train_val_pred.values, title="Act./Pr. plot: "+model_label+"\nTest set", xtext="Actual pIC50", ytext="Predicted pIC50")
    fig.text(0.15, 0.8, "R2 score: "+str(np.round(r2s, 3)), size=16)
    fig.text(0.15, 0.75, "MSE: "+str(np.round(mse, 3)), size=16)
    fig.savefig(model_label+"_a_vs_p_test.pdf", dpi=300, format="pdf")    
    
    entry_line = model_label+"_test"+SEP+str(r2s)+SEP+str(mse)+SEP+str(msle)+SEP+str(mae)+SEP+str(mape)+"\n"
    if result_file: fle.write(entry_line)
    
    l.info("R2 score on test: %s", r2s)
    l.info("MSE score on test: %s", mse)
    l.info("MSLE score on test: %s", mse)
    l.info("MAE score on test: %s", mae)
    l.info("MAPE score on test: %s", mape)

    smiles_codes_val = m_r.Data["smiles_codes_val"]
    smiles_codes_train = m_r.Data["smiles_codes_train"]
    m_r.Pipeline.fit(X=smiles_codes_train, y=None)
    X_train = create_molecular_features(pipeline=m_r.Pipeline, smiles_codes=smiles_codes_train)
    X_train = process_molecular_features(pipeline=m_r.Pipeline, X=X_train)
    y_train_pred = m_r.Predict(X_train)
    y_train = m_r.Data["targets_train"]

    
    r2s = r2_score(y_train, y_train_pred)
    mse = mean_squared_error(y_train, y_train_pred)
    msle = mean_squared_log_error(y_train, y_train_pred)
    mae = mean_absolute_error(y_train, y_train_pred)
    mape = mean_absolute_percentage_error(y_train, y_train_pred)
    
    _, fig = plotScatterWithIdeal(y_train.values, y_train_pred.values, "Act./Pr. plot: "+model_label+"\nTraining set", xtext="Actual pIC50", ytext="Predicted pIC50")
    fig.text(0.15, 0.8, "R2 score: "+str(np.round(r2s, 3)), size=16)
    fig.text(0.15, 0.75, "MSE: "+str(np.round(mse, 3)), size=16)
    fig.savefig(model_label+"_a_vs_p_training.pdf", dpi=300, format="pdf")
    
    entry_line = model_label+"_test"+SEP+str(r2s)+SEP+str(mse)+SEP+str(msle)+SEP+str(mae)+SEP+str(mape)+"\n"
    if result_file: fle.write(entry_line)
    
    l.info("R2 score on training: %s", r2s)
    l.info("MSE score on training: %s", mse)
    l.info("MSLE score on training: %s", msle)
    l.info("MAE score on training: %s", mae)
    l.info("MAPE score on training: %s", mape)
    
    if result_file: fle.close()
    
    
def get_model_predictions(training_configuration_file, smiles_codes_ext=None):

    with open(training_configuration_file.as_posix(), "r") as f:
        training_params = json.load(f)
    
    model = mw.M_XGBoost(prediction_type=mw.Prediction_Type.REGRESSION, pipeline_configuration=None)
    model_file = training_params["model_storage"]["resulting_model"]
    model.Load(model_file)

    prediction_methodology = training_params["prediction_methodology"]
    prediction_type=training_params["prediction_type"]

    if prediction_methodology == mw.Prediction_Methodology.MLP:
        model = mw.M_MLP(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.XGBOOST:
        model = mw.M_XGBoost(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.RF:
        model = mw.M_RandomForest(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.SVM:
        model = mw.M_SVM(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.RIDGE:
        model = mw.M_Ridge(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.BAGGING:
        model = mw.M_Bagging(prediction_type=prediction_type, pipeline_configuration=None)
    else:
        raise NotImplementedError

    model_label = prediction_methodology + "_" + create_model_label(training_params)

    model_file = training_params["model_storage"]["resulting_model"]
    model.Load(model_file)
    l.info("Optimal hyperparameters:, %s", str(model.Hyperparameters))
    smiles_codes_val = model.Data["smiles_codes_val"]
    smiles_codes_train = model.Data["smiles_codes_train"]
    model.Pipeline.fit(X=smiles_codes_train, y=None)
    
    
    X_val = create_molecular_features(pipeline=model.Pipeline, smiles_codes=smiles_codes_val)
    X_val = process_molecular_features(pipeline=model.Pipeline, X=X_val)
    y_train_val_pred = model.Predict(X_val)
    y_train_val = model.Data["targets_val"]
    
    X_train = create_molecular_features(pipeline=model.Pipeline, smiles_codes=smiles_codes_train)
    X_train = process_molecular_features(pipeline=model.Pipeline, X=X_train)
    y_train_pred = model.Predict(X_train)
    y_train = model.Data["targets_train"]
    
    if type(smiles_codes_ext) == pd.Series and len(smiles_codes_ext) > 0:
        X_ext = create_molecular_features(pipeline=model.Pipeline, smiles_codes=smiles_codes_ext)
        X_ext = process_molecular_features(pipeline=model.Pipeline, X=X_ext)
        y_ext_pred = model.Predict(X_ext)
    else:
        X_ext = None
        y_ext_pred = None
        
    return X_val, y_train_val, y_train_val_pred, X_train, y_train, y_train_pred, X_ext, y_ext_pred


def get_model_predictions(training_configuration_file, smiles_codes_ext=None):

    with open(training_configuration_file.as_posix(), "r") as f:
        training_params = json.load(f)
    
    #model = mw.M_XGBoost(prediction_type=mw.Prediction_Type.REGRESSION, pipeline_configuration=None)
    #model_file = (co.MODEL_DIR/training_params["model_storage"]["resulting_model"]).as_posix()
    #model.Load(model_file)

    prediction_methodology = training_params["prediction_methodology"]
    prediction_type=training_params["prediction_type"]

    if prediction_methodology == mw.Prediction_Methodology.MLP:
        model = mw.M_MLP(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.XGBOOST:
        model = mw.M_XGBoost(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.RF:
        model = mw.M_RandomForest(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.SVM:
        model = mw.M_SVM(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.RIDGE:
        model = mw.M_Ridge(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.BAGGING:
        model = mw.M_Bagging(prediction_type=prediction_type, pipeline_configuration=None)
    else:
        raise NotImplementedError

    model_label = prediction_methodology + "_" + create_model_label(training_params)

    model_file = training_params["model_storage"]["resulting_model"]
    model.Load(model_file)
    l.info("Optimal hyperparameters:, %s", str(model.Hyperparameters))
    #smiles_codes_val = model.Data["smiles_codes_val"]
    smiles_codes_train = model.Data["smiles_codes_train"]
    model.Pipeline.fit(X=smiles_codes_train, y=None)
    
    X_val = create_molecular_features(pipeline=model.Pipeline, smiles_codes=smiles_codes_val)
    X_val = process_molecular_features(pipeline=model.Pipeline, X=X_val)
    y_train_val_pred = model.Predict(X_val)
    y_train_val = model.Data["targets_val"]
    
    X_train = create_molecular_features(pipeline=model.Pipeline, smiles_codes=smiles_codes_train)
    X_train = process_molecular_features(pipeline=model.Pipeline, X=X_train)
    y_train_pred = model.Predict(X_train)
    y_train = model.Data["targets_train"]
    
    if type(smiles_codes_ext) == pd.Series and len(smiles_codes_ext) > 0:
        X_ext = create_molecular_features(pipeline=model.Pipeline, smiles_codes=smiles_codes_ext)
        X_ext = process_molecular_features(pipeline=model.Pipeline, X=X_ext)
        y_ext_pred = model.Predict(X_ext)
    else:
        X_ext = None
        y_ext_pred = None
        
    return X_val, y_train_val, y_train_val_pred, X_train, y_train, y_train_pred, X_ext, y_ext_pred

def get_model_ext_predictions(training_configuration_file, smiles_codes=None):

    with open(training_configuration_file.as_posix(), "r") as f:
        training_params = json.load(f)

    prediction_methodology = training_params["prediction_methodology"]
    prediction_type=training_params["prediction_type"]

    if prediction_methodology == mw.Prediction_Methodology.MLP:
        model = mw.M_MLP(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.XGBOOST:
        model = mw.M_XGBoost(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.RF:
        model = mw.M_RandomForest(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.SVM:
        model = mw.M_SVM(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.RIDGE:
        model = mw.M_Ridge(prediction_type=prediction_type, pipeline_configuration=None)
    elif prediction_methodology == mw.Prediction_Methodology.BAGGING:
        model = mw.M_Bagging(prediction_type=prediction_type, pipeline_configuration=None)
    else:
        raise NotImplementedError

    model_label = prediction_methodology + "_" + create_model_label(training_params)

    model_file = training_params["model_storage"]["resulting_model"]
    model.Load(model_file)
    l.info("Optimal hyperparameters:, %s", str(model.Hyperparameters))
    smiles_codes_train = model.Data["smiles_codes_train"]
    model.Pipeline.fit(X=smiles_codes_train, y=None)
    
    if type(smiles_codes) == pd.Series and len(smiles_codes) > 0:
        X_ext = create_molecular_features(pipeline=model.Pipeline, smiles_codes=smiles_codes)
        X_ext = process_molecular_features(pipeline=model.Pipeline, X=X_ext)
        y_ext_pred = model.Predict(X_ext)
        
    return y_ext_pred