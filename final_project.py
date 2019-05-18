# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:27:57 2019

@author: mukes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.linear_model import LinearRegression
# ensemble models
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
#from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import operator
from sklearn.model_selection import ShuffleSplit, GridSearchCV, KFold
from collections import defaultdict
from sklearn.feature_selection import RFE
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.base import BaseEstimator

def ts_normalize(df, columns):
    result = df.copy()
    sc = StandardScaler()
    #scale variables along time in the same security
    for col in columns:
        result[col] = sc.fit_transform(np.array(result[col]).reshape(-1,1))  
    return result

def ts_fillna(df, column, default_value=0):
    result = df.copy()
    #backfill missing data, try the previous value, if not existing use default 
    na_rows = result[column].isna().sum()
    if na_rows > 0:
        result.vol.fillna(method='ffill', inplace=True)    
        result.vol.fillna(default_value, inplace=True)          
    return result

def get_weight_ffd(d, thres, lim):
    w, k = [1.], 1
    ctr = 0
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
        ctr += 1
        if ctr == lim - 1:
            break
        
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

def frac_diff_ffd(x, d, thres=1e-3):
    w = get_weight_ffd(d, thres, len(x))
    width = len(w) - 1
    output = []
    output.extend([0] * width)
    for i in range(width, len(x)):
        output.append(np.dot(w.T, x[i - width:i + 1])[0])
    
    return np.array(output)

def add_vol_ffd(series, fd):
    series['vol_ffd'] = frac_diff_ffd(series.vol, fd)    

def get(df):
    #X = df[['vol', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']].values
    X = df[['vol_ffd', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']].values
    y = df[['fut_ret']].values.flatten()
    return (X, y)

#########################################################
# Get Data
#########################################################    
def get_data():
    df = pd.read_csv('dat_final.csv')
    df.set_index('Date')
    df = df.loc[df['Date']==0]
    return df

#########################################################
# Pre process data
#########################################################
def pre_process(df, input_vars, normalize=True):
    # Check the statistics of the columns of the merged dataframe and check for outliers
    print(df.describe())

    # plot histogram
#    df.hist(sharex = False, sharey = False, xlabelsize = 4, ylabelsize = 4, figsize=(10, 10))
#    plt.show()

    #normalize data and fill empty records
    median = df['vol'].median()
    df = ts_fillna(df, 'vol', median)
    
    if normalize:
        df = ts_normalize(df, input_vars) 
  
    # factor of differentiation decided as the one that makes all the samples in the dataset stationary
    add_vol_ffd(df, 0.7) 
    return df

#########################################################
# Clean data
# 1. fill zero/null vol value with median
# 2. get rid of the securities with less than empty_thres % valid return or vol data (zeros)
#########################################################
def clean_data(org_data, empty_thres=0.25):      
    org_data.drop('Time',axis=1,inplace=True)
    # looks like vol=0 is a data capture error, will fill with previous 
    # value/dataset median
    org_data.vol.replace(0,np.nan,inplace=True)
  
    # extract time-series per security
    sec_ids = org_data['sec_id'].unique()
    series = [org_data[org_data['sec_id']== i] for i in sec_ids]
    series = [ x.set_index('Date') for x in series] #sets the index to the Date
  
    # get rid of the securities with less than empty_thres % valid return or vol data (zeros)
    empty_secs = set()
    for i in range(len(series)):
        s = series[i]
        total = s.shape[0]
        zeros = len(s[s.fut_ret==0.0])
        if zeros>total*empty_thres:
            empty_secs.add(s.sec_id[0])

    discarded = len(empty_secs)
    print(f"Discarding {discarded} securities with incomplete return data (threshold={empty_thres})")
    clean_series = []
    for s in series:
        if s.sec_id[0] not in empty_secs:
            clean_series.append(s)
  
    return pd.concat(clean_series)
    
#########################################################
# List of all the models
#########################################################        
def get_model(name):
    # vanilla models 
    if name == 'LinearRegression': return LinearRegression()
    elif name == 'RandomForestRegressor': return RandomForestRegressor(n_estimators=10,min_samples_split=0.001)
    elif name == 'KNeighborsRegressor': return KNeighborsRegressor()
    elif name == 'GradientBoostingRegressor': 
        return GradientBoostingRegressor(learning_rate=0.1,n_estimators=20,max_depth=2,
                                         max_features=6,min_samples_split=4000,min_samples_leaf=200)
    elif name == 'LSTMRegressor': return LSTMRegressor(time_step=30)
    # models with cross validations
    elif name == 'GradientBoosterWithKFold': 
        return CrossValidationModelRegressor('GradientBoostingRegressor', params={
            'learning_rate':[0.1, 0.01],
            'n_estimators':[20, 100],
            'max_depth':[3, 5],
            'max_features':[6, 'sqrt'],
            'min_samples_split':[4000, 0.001],
            'min_samples_leaf':[200]
            }, n_splits=5)
    elif name == 'GradientBoostingRegressor_BEST':
        #from gb_param_tuning
        return GradientBoostingRegressor(learning_rate=0.1,n_estimators=100,max_depth=3,
                                         max_features='sqrt',min_samples_split=0.001)
    # cluster based models
    elif name == 'ClusterLinearRegressor': return ClusterRegressor('LinearRegression')
    elif name == 'ClusterGradientBoostingRegressor': return ClusterRegressor('GradientBoostingRegressor', params={
            'learning_rate':0.1,'n_estimators':20, 'max_depth':2,
            'max_features':6,'min_samples_split':4000, 'min_samples_leaf':200
            })
    elif name == 'ClusterLSTMRegressor': return ClusterRegressor('LSTMRegressor', params={'time_step':1})
    else: return LinearRegression()
  
#########################################################
# Regressor with Cross Validation (K-fold)
#########################################################
class CrossValidationModelRegressor:
    def __init__(self, model_name, params, n_splits):
        self.model_name = model_name
        self.params = params
        self.n_splits = n_splits
        
    def fit(self, X, y):
        constructor = globals()[self.model_name]
        estimator = constructor()

        cv = KFold(n_splits=self.n_splits)
        classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=self.params, n_jobs=4)
        classifier.fit(X, y)
        
        self.classifier = classifier
        
        best_est = classifier.best_estimator_
        print('best estimator: {}'.format(best_est))
        best_estimator = globals()[self.model_name]
        best_estimator = best_estimator.set_params(best_est)
        
        best_estimator.fit(X, y)
        self.best_estimator = best_estimator
        
    def predict(self, X_test):
        return self.best_estimator.predict(X_test)
 
#########################################################
# LSTM
#########################################################
class LSTMRegressor():
    def __init__(self, time_step=30):
        self.time_step = time_step
        
    def set_params(self, params):
        print(params)
        self.time_step = params['time_step']
        return self;
    
    def build_timeseries(self, X, y):
        time_step = self.time_step
        d1 = X.shape[0] - self.time_step + 1
        d2 = X.shape[1]
        
        X_m = np.zeros((d1, time_step, d2))
        y_m = np.zeros((d1, time_step, ))
        
        for i in range(d1):
            X_m[i] = X[i:time_step+i]
            y_m[i] = y[i:time_step+i] 
        
        return (X_m, y_m)
    
    def fit(self, X, y):
        (X_m, y_m) = self.build_timeseries(X, y)
        
        batch_size = 10000
        regressor = Sequential()

        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_m.shape[1], X_m.shape[2])))
        regressor.add(Dropout(0.2))
        
        # Adding a second LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        # Adding a third LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        # Adding a fourth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = 50))
        regressor.add(Dropout(0.2))
        
        # Adding the output layer
        regressor.add(Dense(units=self.time_step))
        # Compiling the RNN
        regressor.compile(optimizer = 'ADAgrad', loss = 'mean_squared_error')
        
        # Fitting the RNN to the Training set
        regressor.fit(X_m, y_m, epochs =1, batch_size = batch_size)
        self.regressor = regressor
        
    def score(self, X_test, y_test):
        (X_test_m, y_test_m) = self.build_timeseries(X_test, y_test)
        y_predict = self.regressor.predict(X_test_m)
        return r2_score(y_test_m, y_predict)

    def predict(self, X_test):
        y = np.zeros(X_test.shape[0]) # dummy
        (X_test_m, y_test_m) = self.build_timeseries(X_test, y)
        y_predict = self.regressor.predict(X_test_m)
        return y_predict
   
#########################################################
# Regressor using clustering 
# 1. divides data into clusters using K-means
# 2. regression is run for each cluster
#########################################################     
class ClusterRegressor:
    def __init__(self, model_name, params):
        self.model_name = model_name
        self.params = params
        
    def kmeans_cluster(self, X):
        sse = []
        clusters = range(2,15,2) 
        for k in clusters:
            print('k:{}'.format(k))
            kmeans = KMeans(n_clusters = k)
            kmeans.fit(X)
            sse.append(kmeans.inertia_) #SSE for each n_clusters
            
        plt.plot(clusters, sse)
        plt.title("Elbow Curve")
        plt.show()
        
        k = 8  # clusters to use
        kmeans = KMeans(n_clusters = k).fit(X)
        plt.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
        plt.show()    
        self.kmeans = kmeans;
    
    def fit(self, X, y):
        self.kmeans_cluster(X)
        labels = self.kmeans.labels_

        clusters = pd.DataFrame(labels, columns=['cluster'])
        unique_clusters = clusters['cluster'].unique()
        
        df_X = pd.DataFrame(X)
        df_y = pd.DataFrame(y)
        
        df_X = pd.concat([df_X, clusters], axis=1)
        df_y = pd.concat([df_y, clusters], axis=1)
        
        models = {}
        for cluster in unique_clusters:
            filtered_df_X = df_X.loc[df_X['cluster']==cluster]
            filtered_df_y = df_y.loc[df_y['cluster']==cluster]
            
            X_train = filtered_df_X.drop(columns='cluster').values
            y_train = filtered_df_y.drop(columns='cluster').values.flatten()
            
            regressor = globals()[self.model_name]()
            regressor = regressor.set_params(self.params)
            print('cluster:{}, model:{}, data size:{}'.format(cluster, self.model_name, len(y_train)))
            regressor.fit(X_train, y_train)
            models[cluster] = regressor
             
        print('fit complete')
        self.models = models
        
    def predict(self, X_test):
        labels = self.kmeans.predict(X_test)       
        clusters = pd.DataFrame(labels, columns=['cluster'])
        df = pd.DataFrame(X_test)
        
        df = pd.concat([df, clusters], axis=1)
        y_pred_arr =[]
        print('start predict')
        for index, row in df.iterrows():
            r = np.reshape(np.array(row[:-1]), (1, len(row)-1))
            y_pred = self.models[row['cluster']].predict(r)
            y_pred_arr.append(y_pred[0])
        
        print('finished predict')
        return y_pred_arr


def featureselection_rfe(X, y):     
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X2 = poly.fit_transform(X)
    
    model=LinearRegression()
    #discard half of the features
    rfe = RFE(model)
    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=1/6, shuffle=False)
    rfe = rfe.fit(X_train, y_train)
    names=poly.get_feature_names()

    print ("Features by RFE process:")
    print (sorted(zip(map(lambda x: x, rfe.support_), 
                  names), reverse=True))
    print(rfe.support_)
    print(rfe.ranking_)
    return X2[:,rfe.ranking_==1]

def featureselection_mda(X, y):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X2 = poly.fit_transform(X)
    
    scores = defaultdict(list)
    features = poly.get_feature_names()

    reg = LinearRegression()
    count = 1
    splits = 100
    for train_idx, test_idx in ShuffleSplit(n_splits=splits).split(X2):
        X_train, X_test = X2[train_idx], X2[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        rf = reg.fit(X_train, y_train)
        acc = r2_score(y_test, rf.predict(X_test))
        if count%10==0 : print(f"run {count}/{splits}")
        for i in range(X2.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = r2_score(y_test, rf.predict(X_t))
            scores[features[i]].append((acc-shuff_acc)/acc)
        count = count + 1

    mda_features = [f for f in scores.keys()]
    mda_importance = [(np.mean(score)) for score in scores.values()] 
    mda_indices = np.argsort(mda_importance)[::-1]
    
    print ("Features by MDA process:")
    
    rcParams['figure.figsize'] = 16, 20
    plt.title('Feature Importances by MDA')
    plt.barh(range(len(mda_indices)), [mda_importance[i] for i in mda_indices], color='#8f63f4', align='center')
    plt.yticks(range(len(mda_indices)), [mda_features[i] for i in mda_indices])
    plt.xlabel('Mean decrease accuracy')
    plt.show()
    
    
    features = dict(zip(mda_features,mda_importance))
    sorted_feats = sorted(features.items(), key=operator.itemgetter(1), reverse=True)
    print(pd.DataFrame(sorted_feats))
    # discard half of the features
    return X2[:, mda_indices[0:X2.shape[1]//2]]

def gb_param_tuning(X, y, cv_splits=5):
    from sklearn.model_selection import GridSearchCV
    parameters = {'learning_rate':[0.1, 0.01, 0.001], 
                  'n_estimators':[20, 50, 100],
                  'min_samples_split': [0.02, 0.01, 0.001],
                  'max_depth' : [2, 3, 4],
                  'max_features': [0.25, 'sqrt']}  #0.25 ~ 10, sqrt ~ 6
                  
    gb = GradientBoostingRegressor()
    clf = GridSearchCV(gb, parameters, cv=cv_splits, iid=False, verbose=10, n_jobs=4)
    clf.fit(X, y)
    return clf.best_params_

#########################################################
# Run Model
#########################################################   
def run(selection='mda'):
    input_vars = ['vol','X1','X2','X3','X4','X5','X6','X7']
    df = get_data()
    df = clean_data(df)
    df = pre_process(df, input_vars)
    
    (X, y) = get(df)
    if selection == 'mda':
        X2 = featureselection_mda(X, y)
    elif selection == 'rfe':
        X2 = featureselection_rfe(X, y)
    elif selection == 'all': #all features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X2 = poly.fit_transform(X)
    else: #only basic features
        X2 = X
          
    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=1/6, shuffle=False)
    # Append the models to the models list
    model_names = ['LinearRegression', 'GradientBoostingRegressor', 'RandomForestRegressor', 
                   'KNeighborsRegressor', 'ClusterLinearRegressor', 'ClusterGradientBoostingRegressor']
    model_names = ['LSTMRegressor']
    model_names = ['LinearRegression', 'GradientBoostingRegressor', 'GradientBoostingRegressor_100']
    model_names = ['GradientBoosterWithKFold', 'GradientBoostingRegressor_100']
    model_names = ['ClusterLSTMRegressor', 'LSTMRegressor']
    # fit
    result = {}
    for name in model_names:
        print('Running model:{}'.format(name))
        cr = get_model(name)
        cr.fit(X_train, y_train)
        
        if name != 'LSTMRegressor': 
            r2_out = r2_score(y_test, cr.predict(X_test))
            r2_in = r2_score(y_train, cr.predict(X_train))
        else:
            r2_out = cr.score(X_test, y_test)
            r2_in = cr.score(X_train, y_train)
        
        result[name] = (r2_in, r2_out)
        print('{} r2_in : {}, r2_out:{}'.format(name, r2_in, r2_out))

run('rfe')

