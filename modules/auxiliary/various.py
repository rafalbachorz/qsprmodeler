import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, precision_score, r2_score, precision_recall_curve, roc_curve, roc_auc_score, average_precision_score, precision_score, recall_score, \
ConfusionMatrixDisplay, f1_score, accuracy_score
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd
import tensorflow as tf
import random as rn
import json
from auxiliary import loggers

l = loggers.get_logger(logger_name="logger")

def boxplot(data, title, quantity_name, to_file=False, file_name=None):
    fig1, ax1 = plt.subplots(figsize=(10, 8), facecolor='white')
    ax1.set_title(title, fontsize=18)
    ax1.boxplot(data)
    plt.xticks([1], [quantity_name], fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel(quantity_name, fontsize=16)
    plt.grid(True)
    if to_file and file_name:
        plt.savefig(file_name, format="tiff", transparent=False, dpi=200)
    return ax1

def histogram(data, title, quantity_name, n_bins=15, to_file=False, file_name=None):
    fig1, ax1 = plt.subplots(figsize=(10, 8), facecolor='white')
    ax1.set_title(title, fontsize=18)
    ax1.hist(data, 50)
    plt.yticks(fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.xlabel(quantity_name, fontsize=16)
    plt.grid(True)
    if to_file and file_name:
        plt.savefig(file_name, format="tiff", transparent=False, dpi=200)
    return ax1

def plotScatterWithIdeal(x, y, title, confidence=None, xtext="Actual", ytext="Predicted"):
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    plt.scatter(x, y, c=confidence, cmap="gray")
    p1 = max(max(y), max(x)) + 1.0
    p2 = min(min(y), min(x)) - 1.0
    plt.plot([p1, p2], [p1, p2], 'k-')
    plt.grid()
    ax.set_title(title, fontsize=18)
    ax = plt.gca()
    ax.set_xlabel(xtext, fontsize=16)
    ax.set_ylabel(ytext, fontsize=16)
    return ax, fig

def create_precision_recall_plot(y_test, y_pred, title="Precision-recall plot", id_plot=1):
    plot1 = plt.figure(id_plot)
    a, b, _ = precision_recall_curve(y_test, y_pred)
    plt.plot(b, a)
    plt.grid()
    plt.fill_between(b, 0, a, alpha=0.5)
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    rocauc = roc_auc_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_pred)
    plt.text(0.15, 0.7, "ROCAUC = " + "{0:.3f}".format(rocauc), fontsize=14,
             verticalalignment="top", bbox=props)
    plt.text(0.15, 0.55, "AP = " + "{0:.3f}".format(ap), fontsize=14,
             verticalalignment="top", bbox=props)
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    return plt.figure

def create_roc_plot(y_test, y_pred, title="ROC plot", id_plot=1):
    plot1 = plt.figure(id_plot)
    a, b, _ = roc_curve(y_test, y_pred)
    plt.plot(a, b)
    plt.grid()
    plt.fill_between(a, 0, b, alpha=0.5)
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    rocauc = roc_auc_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_pred)
    plt.text(0.15, 0.7, "ROCAUC = " + "{0:.3f}".format(rocauc), fontsize=14,
             verticalalignment="top", bbox=props)
    plt.text(0.15, 0.55, "AP = " + "{0:.3f}".format(ap), fontsize=14,
             verticalalignment="top", bbox=props)
    plt.title(title)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    return plt.figure

def evaluate_regressor_quality(X, y, hps, provide_regressor=False):
    regressor = XGBRegressor(use_label_encoder=False, eval_metric="rmse", tree_method="gpu_hist", **hps)
    scoring = {"mae": make_scorer(mean_absolute_error), "mse": make_scorer(mean_squared_error), "r2": make_scorer(r2_score)}
    scores = cross_validate(regressor, X, y, cv=4, scoring=scoring)
    if provide_regressor:
    # after getting crossvalidated measures, train on entire set
        regressor = XGBRegressor(use_label_encoder=False, eval_metric="rmse", tree_method="gpu_hist", kwargs=hps)
        regressor.fit(X, y)
    else:
        regressor = None
    return scores, regressor

def create_index_dictionaries(raw_index):
    raw_to_ordered = {}
    ordered_to_raw = {}
    ordered_index = range(len(raw_index))
    for raw, ordered in zip(raw_index, ordered_index):
        raw_to_ordered[raw] = ordered
        ordered_to_raw[ordered] = raw
    return raw_to_ordered, ordered_to_raw

def regression_scores(estimator, X, y):
    y_pred = estimator.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mae, mse, r2

def initialize_seeds(random_seed):
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    rn.seed(random_seed)

def create_crisp_index(y_num, threshold, column_name="IC50_class"):
    y_crisp = pd.DataFrame(y_num > threshold)
    y_crisp = y_crisp.astype(int)
    y_crisp = y_crisp.set_index(y_num.index)
    y_crisp.columns = [column_name]
    return y_crisp

