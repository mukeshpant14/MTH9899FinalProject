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
from sklearn import linear_model
import pickle
from time import gmtime, strftime

def mad_filter(samples, n=5):
    result = samples.copy()
    center = np.median(result)
    mad = np.median(abs(result-center))
    minimum = center - n*mad
    maximum = center + n*mad
    for i in range(len(result)):
        val = result[i]
        if val < minimum:
            result[i] = minimum
        elif  val > maximum:
            result[i] = maximum
    return result

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

def plot_prediction(keys, y_test, y_pred):
    keys_df = pd.DataFrame(keys, columns =['Date', 'sec_id'])
    y_test_df = pd.DataFrame(y_test, columns =['y_test'])
    y_pred_df = pd.DataFrame(y_pred, columns =['y_pred'])
    
    print('in plot')
    print(keys_df.shape)
    print(y_test_df.shape)
    print(y_pred_df.shape)
    
    df = pd.concat([keys_df, y_test_df, y_pred_df], axis=1, join_axes=[keys_df.index])
    print(df.shape)
    
    df = df.loc[df['sec_id'] == 0]
  
    df.set_index('Date')
    df.plot(kind='line',x='Date', y='y_test', ax=plt.gca())
    df.plot(kind='line',x='Date', y='y_pred', color='red', ax=plt.gca())
    
    plt.show()

def get_final_oos_data():
    print('Reading from file dat_final_oos_noy.csv')
    df =pd.read_csv('dat_final_oos_noy.csv')
    df.set_index('Date')
    df = df.loc[df['sec_id'].isin([0, 1])]
    return df
        
#########################################################
# Get Data
#########################################################    
def get_data():  
    print('Reading from file dat_final.csv')
    df = pd.read_csv('dat_final.csv')    
    #    df = df.loc[df['Date'].isin([0, 1])]
#    df = df.loc[df['sec_id'].isin([0, 1, 2])]  # should remove these only for testing

    df.set_index('Date')
    return df

##########################################################
## Pre process data
##########################################################
#def pre_process(df, input_vars, normalize=True):
#    # Check the statistics of the columns of the merged dataframe and check for outliers
#    print(df.describe())
#
#    # plot histogram
##    df.hist(sharex = False, sharey = False, xlabelsize = 4, ylabelsize = 4, figsize=(10, 10))
##    plt.show()
#
#    #normalize data and fill empty records
#    print('Done Preprocessing data...')
#    return df

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
    clean_series = []
    if empty_thres < 1:
        for i in range(len(series)):
            s = series[i]
            total = s.shape[0]
            zeros = len(s[s.fut_ret==0.0])
            if zeros>total*empty_thres:
                empty_secs.add(s.sec_id[0])

        discarded = len(empty_secs)
        print(f"Discarding {discarded} securities with incomplete return data (threshold={empty_thres})")
        
        for s in series:
            if s.sec_id[0] not in empty_secs:
                clean_series.append(s)
    else:
        for s in series:
            clean_series.append(s)
     
    #normalize data and fill empty records
    median = org_data['vol'].median()
    input_vars = ['vol','X1','X2','X3','X4','X5','X6','X7']
    clean_series = [ ts_fillna(x, 'vol', median) for x in clean_series]
    
    clean_series = [ ts_normalize(x, input_vars) for x in clean_series] 
    
    #make vol a stationary variable using fractional differencing
    #the differencing factor is decided as the value that makes more than 95% 
    #of the samples in the dataset stationary
#    print(clean_series)
    [add_vol_ffd(x, 0.7) for x in clean_series]
      
    print('Done cleaning data...')
    d = pd.concat(clean_series)
    return d
    
#########################################################
# List of all the models
#########################################################        
def get_model(name):
    # vanilla models 
    if name == 'LinearRegression': 
        return LinearRegression()
    elif name == 'RandomForestRegressor': 
        return RandomForestRegressor(n_estimators=10,min_samples_split=0.001)
    elif name == 'KNeighborsRegressor': 
        return KNeighborsRegressor()
    elif name == 'GradientBoostingRegressor': 
        return GradientBoostingRegressor(learning_rate=0.1,n_estimators=20,max_depth=2,
                                         max_features=6,min_samples_split=4000,min_samples_leaf=200)
    elif name == 'LSTMRegressor': 
        return LSTMRegressor(time_step=1)
    elif name == 'LassoCV':
        return linear_model.LassoCV(normalize=True, cv=5) 
    #----------------------------------------------
    # cluster based models ClusterLinearRegressor
    #----------------------------------------------
    elif name == 'ClusterLinearRegressor': return ClusterRegressor('LinearRegression', params={'n_jobs':1})
    elif name == 'ClusterGradientBoostingRegressor': return ClusterRegressor('GradientBoostingRegressor', params={
            'learning_rate':0.1,'n_estimators':100, 'max_depth':3,
            'max_features':'sqrt','min_samples_split':0.001, 'min_samples_leaf':200
            })
    elif name == 'ClusterLSTMRegressor': return ClusterRegressor('LSTMRegressor', params={'time_step':1})
    #----------------------------------------------
    # Best models 
    #----------------------------------------------
    elif name == 'GradientBoostingRegressor_BEST':
        #from CrossValidationModelRegressor
        return GradientBoostingRegressor(learning_rate=0.1,n_estimators=100,max_depth=3,
                                         max_features='sqrt',min_samples_split=0.001)
    elif name == 'ClusterGradientBoostingRegressor_BEST': 
        return ClusterRegressor('GradientBoostingRegressor', params={
            'learning_rate':0.1,'n_estimators':100, 'max_depth':3,
            'max_features':'sqrt','min_samples_split':0.001, 'min_samples_leaf':200
            })
    elif name == 'RandomForestRegressor_BEST':
        #from CrossValidationModelRegressor
        return RandomForestRegressor(n_estimators=100,max_depth=5,max_features=6,
                                     min_samples_split=0.001,min_samples_leaf=0.0001)
    #----------------------------------------------
    # Models for parameter tuning
    #----------------------------------------------
    elif name == 'ForestWithKFold': 
        return CrossValidationModelRegressor('RandomForestRegressor', params={
            'n_estimators':[20, 100],
            'max_depth':[3, 5, None],
            'max_features':[0.5, 10, 6],
            'min_samples_split':[0.001, 0.0001],
            'min_samples_leaf':[0.001, 0.0001],
            'n_jobs':[4]
            }, n_splits=5)
    elif name == 'GradientBoosterWithKFold': 
        return CrossValidationModelRegressor('GradientBoostingRegressor', params={
            'learning_rate':[0.1, 0.01],
            'n_estimators':[20, 100],
            'max_depth':[3, 5],
            'max_features':[6, 'sqrt'],
            'min_samples_split':[4000, 0.001],
            'min_samples_leaf':[200]
            }, n_splits=5)
    else: raise Exception('Invalid model: {}'.format(name))
    
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
        classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=self.params, n_jobs=4, verbose=10)
        classifier.fit(X, y)
        
        self.classifier = classifier
        
        best_est = classifier.best_estimator_
        print('best estimator: {}'.format(best_est))
        best_estimator = globals()[self.model_name]
        best_estimator = best_estimator.set_params(**best_est)
        
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
        self.cols = ['vol_ffd', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']
        
    def set_params(self, **params):
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
        
        batch_size = 1000
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
        regressor.add(Dense(units= self.time_step))

        # Adding the output layer
#        regressor.add(Dense(units = 1))
        
        # Compiling the RNN
        regressor.compile(optimizer = 'ADAgrad', loss = 'mean_squared_error')
        
        # Fitting the RNN to the Training set
        regressor.fit(X_m, y_m, epochs=5, batch_size = batch_size)
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
        self.cluster_feature_index = [0, 1]
       
    def get_cluster_feature_data(self, X):
        # select data for specific feature
        df_X = pd.DataFrame(X)[self.cluster_feature_index] # feature
        X_feat = df_X.values
        return X_feat
    
    def kmeans_cluster(self, X):
        X_feat = self.get_cluster_feature_data(X)
        
        sse = []
        clusters = range(2,15,2) 
        for k in clusters:
            print('k:{}'.format(k))
            kmeans = KMeans(n_clusters = k)
            kmeans.fit(X_feat)
            sse.append(kmeans.inertia_) #SSE for each n_clusters
            
        plt.plot(clusters, sse)
        plt.title("Elbow Curve")
        plt.show()
        
        print(kmeans)
        k = 8  # clusters to use
        kmeans = KMeans(n_clusters = k).fit(X_feat)
        plt.scatter(X_feat[:,0],X_feat[:,1], c = kmeans.labels_, cmap ="rainbow")
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
#            print(regressor)
#            print('parameters available: {}'.format(regressor.get_params().keys()))
#            print(**self.params)
            regressor = regressor.set_params(**self.params)
            print('cluster:{}, model:{}, data size:{}'.format(cluster, self.model_name, len(y_train)))
            regressor.fit(X_train, y_train)
            models[cluster] = regressor
             
        print('fit complete')
        self.models = models
        
    def predict(self, X_test):
        X_feat = self.get_cluster_feature_data(X_test)
        labels = self.kmeans.predict(X_feat)   
        
        clusters = pd.DataFrame(labels, columns=['cluster'])
        df = pd.DataFrame(X_test)
        
        df = pd.concat([df, clusters], axis=1)
        y_pred_arr =[]
        for index, row in df.iterrows():
            r = np.reshape(np.array(row[:-1]), (1, len(row)-1))
            y_pred = self.models[row['cluster']].predict(r)
            y_pred_arr.append(y_pred[0])
        
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

    reg = linear_model.LassoCV(cv=5)
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
    plt.title('Feature Importances by MDA', fontsize=18)
    plt.barh(range(len(mda_indices)), [mda_importance[i] for i in mda_indices], color='#8f63f4', align='center')
    plt.yticks(range(len(mda_indices)), [mda_features[i] for i in mda_indices], fontsize=16)
    plt.axhline(X2.shape[1]//2, linestyle='dotted')
    plt.xlabel('Mean decrease accuracy', fontsize=18)
    plt.show()
    
    
    features = dict(zip(mda_features,mda_importance))
    sorted_feats = sorted(features.items(), key=operator.itemgetter(1), reverse=True)
    print(pd.DataFrame(sorted_feats))
    # discard half of the features
    print(mda_indices[0:X2.shape[1]//2])
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

def processed_data(selection, final_oos_file=False):   
    after_clean_cols = ['vol_ffd', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']
    
    keys = None
    if final_oos_file:
        y=None # no 'fut_ret' in this case
        df = get_final_oos_data()
        df = clean_data(df, empty_thres=1)
        
        date_df = pd.DataFrame(df.index.values, columns=['Date'])
        sec_df = pd.DataFrame(df['sec_id'].values, columns=['sec_id'])
        keys = pd.concat([date_df, sec_df], axis=1)
        
        X = df[after_clean_cols].values
        
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X2 = poly.fit_transform(X)
        
        if selection=='mda':
            # these indices came from tuning on input data
            indices = [ 1,  9,  0,  3, 29, 20, 43, 22, 18, 28,  6,  4, 11,  7, 41, 38, 39,  2, 10, 34, 14, 12]
            X2 = X2[:, indices]
#        elif selection=='rfe':
#            ranking == []
#            X2 = X2[:,ranking==1]
        else:
            X2 = X
    else:
        df = get_data()
        df = clean_data(df)

        date_df = pd.DataFrame(df.index.values, columns=['Date'])
        sec_df = pd.DataFrame(df['sec_id'].values, columns=['sec_id'])
        keys = pd.concat([date_df, sec_df], axis=1)
        
        X = df[after_clean_cols].values
        y = df[['fut_ret']].values.flatten()
    
        if selection == 'mda':
#            X2 = featureselection_mda(X, y)
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X2 = poly.fit_transform(X)
            indices = [ 1,  9,  0,  3, 29, 20, 43, 22, 18, 28,  6,  4, 11,  7, 41, 38, 39,  2, 10, 34, 14, 12]
            X2 = X2[:, indices]
        elif selection == 'rfe':
            X2 = featureselection_rfe(X, y)
        elif selection == 'all': #all features
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X2 = poly.fit_transform(X)
        else: #only basic features
            X2 = X
    
    print('Done feature selection...')

    #cross-sectional normalization with all features
    sc = StandardScaler()
    for i in range(X2.shape[1]):
        scaled = sc.fit_transform(X2[:,i].reshape(-1,1))
        X2[:,i] = scaled[:,0]
            
    return (X2, y, keys)
    
#########################################################
# Run Model
#########################################################   
def run(selection='mda', model_names=None, save=False, use_saved_model=False):
    (X2, y, keys) = processed_data(selection)
    X_train, X_test, y_train, y_test, keys_train, keys_test = train_test_split(X2, y, keys, test_size=1/6, shuffle=False)
    print(y_test.shape)
    print(keys_test.shape)

#    model_names=[]    
    # fit
    result = {}
    for name in model_names:   
        if use_saved_model:
#            my_models = [f for f in listdir('saved_models/') if f.startswith(name)]
#            if len(my_models) != 1 :
#                raise Exception('found zero or more than one models. {}'.format(my_models))
            my_model = name + "__" + selection + "__20190522" 
            split = my_model.split('__')
            name = split[0]
            feature_selection = split[1]
            date = split[2]
            print(f"Loading model: {name} (created {date}) with ({feature_selection})")
            cr = pickle.load(open('saved_models/'+my_model+".sav", 'rb'))
        else:
            cr = get_model(name)
            print('Running model:{},\n{}'.format(name, cr))
            cr.fit(X_train, y_train)
        
        # if required save to disk
        if save:
            date = strftime("%Y%m%d", gmtime())
            filename = name + "__" + selection + "__" + date + ".sav"
            pickle.dump(cr, open('saved_models/'+filename, 'wb'))
         
        y_test_pred = cr.predict(X_test)
        y_train_pred = cr.predict(X_train)
        
#        if name == 'LSTMRegressor':
#            plot_prediction(keys_test, y_test, y_test_pred)
            
        r2_out = r2_score(y_test, y_test_pred)
        r2_in = r2_score(y_train, y_train_pred)

        result[name] = (r2_in, r2_out)
        print('{} r2_in : {}, r2_out:{}'.format(name, r2_in, r2_out))

    print(result)
    
def run_from_disk(saved_file):
    # load the model from disk
    split = saved_file.split('__')
    name = split[0]
    feature_selection = split[1]
    date = split[2]
    print(f"Loading model: {name} (created {date}) with ({feature_selection})")
    
    cr = pickle.load(open('saved_models/'+saved_file+'.sav', 'rb'))
    (X_test, y_test, keys) = processed_data(feature_selection, final_oos_file=True)
    
    if name != 'LSTMRegressor': 
        y_pred = cr.predict(X_test)
        
        outputfile = name + '__'+ 'mda' + '__' + date + '__' + 'ypredicted.csv'
        df = pd.DataFrame(y_pred, columns = ['y_predicted'])
        df.to_csv('saved_models/'+outputfile, sep='\t')
        print('saved y predicted values in file: {}'.format(outputfile))
    else:
        r2_out = cr.score(X_test, y_test)

    return

# run and save all the models 
#model_names=[
#    'LinearRegression',
#    'RandomForestRegressor_BEST',
#    'GradientBoostingRegressor_BEST',
#    'ClusterLinearRegressor',
#    'ClusterLSTMRegressor',
#    'ClusterGradientBoostingRegressor_BEST',
#]
#run(selection='mda', save=False, use_saved_model=False)
    
#run(selection='none', model_names = ['LSTMRegressor'], save=False, use_saved_model=True)

# Run saved model on OOS data
    
#run_from_disk(saved_file='LinearRegression__mda__20190522')
#run_from_disk(saved_file='RandomForestRegressor_BEST__mda__20190522')
#run_from_disk(saved_file='GradientBoostingRegressor_BEST__mda__20190522')
#run_from_disk(saved_file='ClusterLinearRegressor__mda__20190522')
#run_from_disk(saved_file='ClusterGradientBoostingRegressor_BEST__mda__20190522')
#run_from_disk(saved_file='LSTMRegressor__none__20190522')
