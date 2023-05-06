from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, roc_auc_score, recall_score, \
precision_score, accuracy_score, precision_recall_curve
from sklearn.feature_selection import VarianceThreshold, SequentialFeatureSelector
from sklearn.model_selection import KFold
from sklearn.cluster import  KMeans
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd

import logging

class DataScaling(BaseEstimator, TransformerMixin):
    def __init__(self, scaler, features_all, features_scaling):
        logging.info('Scaler constructor...')
        self.scaler = scaler
        self.features_all = features_all
        self.features_scaling = features_scaling
        self.features_remaining = list(set(features_all) - set(features_scaling))
    
    def fit(self, X, y = None):
        logging.info('Scaler fit...')
        logging.info('I do fit a scaler')

        self.scaler.fit(X[self.features_scaling])
        return self
    
    def transform(self, X, y = None):
        logging.info('Scaler transform...')

        scaled_data = self.scaler.transform(X[self.features_scaling])
        scaled_data = pd.DataFrame(scaled_data, columns=self.features_scaling, index=X.index)
        entire_data = pd.concat([scaled_data, X[self.features_remaining]], axis=1)
        all_features = sorted(entire_data.columns.tolist())
        return entire_data[all_features]
    
    @property
    def get_scaler(self):
        return self.scaler

class DataOHE(BaseEstimator, TransformerMixin):
    def __init__(self, features_all, features_ohe):
        self.features_ohe = features_ohe
        self.features_remaining = list(set(features_all) - set(features_ohe))
        self.ohe = OneHotEncoder()
    
    def fit(self, X, y=None):
        OHE_to_be_fit = X[self.features_ohe]
        self.ohe.fit(OHE_to_be_fit)
        return self
    
    def transform(self, X, y=None):
        logging.info('Entering transform')
        oheFeatures = self.ohe.transform(X[self.features_ohe]).toarray()
        entire_data = pd.concat([X[self.features_remaining], 
                                pd.DataFrame(oheFeatures, index=X.index, columns=self.ohe.get_feature_names())], 
                                axis=1)
        all_features = sorted(entire_data.columns.tolist())
        return entire_data[all_features]

    @property
    def get_ohe(self):
        return self.ohe
    
    @property
    def feature_names(self):
        return self.ohe.get_feature_names()

class DataPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_percent, n_components, random_seed, retain_native, features_ohe, label):
        self.n_percent = n_percent
        self.n_components = n_components
        self.random_seed = random_seed
        self.retain_native = retain_native
        self.features_ohe = features_ohe
        self.label = label
        
    def fit(self, X, y=None):
        logging.info('Fitting PCA')
        fp_columns = [item for item in X.columns.tolist() if item.startswith('fp_')] 
        self.features_numerical = list(set(X.columns.tolist())-set(self.features_ohe)-set(fp_columns))
        self.features_non_numerical = self.features_ohe+fp_columns
        self._pca = self._fit_PCA(X[self.features_numerical])
        return self
        
    def _fit_PCA(self, X):
        pca = PCA(n_components=X.shape[1], random_state=self.random_seed)
        return pca.fit(X)
        
    def transform(self, X, y=None):
        logging.info('Transforming PCA')
        pca_transformed = self._transform_PCA(X[self.features_numerical])
        
        if self.retain_native:
            return pd.concat([pca_transformed, X], axis=1)
        else:
            return pd.concat([pca_transformed, X[self.features_non_numerical]], axis=1)
        
    def _transform_PCA(self, X):
        if self.n_percent is not None:
            pca_cum = self._pca.explained_variance_ratio_.cumsum()
            pca_n = (pca_cum < self.n_percent).sum()
        else:
            pca_n = self.n_components
                
        pca_transformed = self._pca.transform(X)
        pca_transformed = pd.DataFrame(data=pca_transformed[:, :pca_n], index = X.index, 
                                       columns=[self.label+'-pca_'+str(iii) for iii in range(pca_n)])
        return pca_transformed
    
    @property
    def pca(self):
        return self._pca

class DataVarianceThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, threshold, features):
        self.features = features
        self.threshold = threshold
        self.vt = VarianceThreshold(threshold)
    
    def fit(self, X, y=None):
        self.vt.fit(X)
        self.rem_cols = X.columns[self.vt.get_support()]
        return self
    
    def transform(self, X, y=None):
        logging.info('Entering transform')
        other_features = list(set(X.columns.tolist())-set(self.features))
        remained_features = pd.DataFrame(self.vt.transform(X), columns=self.rem_cols, index=X.index)
        
        result = pd.concat([remained_features, X[other_features]], axis=1)
        return result

class DataClusterCreator(BaseEstimator, TransformerMixin):
    def __init__(self, n_cluster, random_seed):
        logging.info('DataClusterCreator constructor...')
        self.n_cluster = n_cluster
        self.random_seed = random_seed
    
    def fit(self, X, y = None):
        logging.info('DataClusterCreator fit...')
        kmeansFitData = X
        logging.info('Here I fit the KMeans, the rows taken: %s', kmeansFitData.shape[0])
        self.kmeans_model = KMeans(n_clusters=self.n_cluster)
        self.kmeans_model.fit(kmeansFitData)
        return self
    
    def transform(self, X, y = None):
        logging.info('DataClusterCreator transform...')
        tmp = self.kmeans_model.predict(X)
        tmp = pd.get_dummies(tmp, prefix='cluster').set_index(X.index)
        X = pd.concat([X, tmp], axis=1)
        return X
    
    @property
    def kmeans(self):
        return self.kmeans_model

