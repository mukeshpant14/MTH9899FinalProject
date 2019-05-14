# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:27:57 2019

@author: mukes
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# neural network
from sklearn.neural_network import MLPClassifier
# ensemble models
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor, RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
#from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

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

def pre_process(df, normalize=False):
    secs = df['sec_id'].unique()
    print('Size of dataset:{}'.format(df.shape[0]))
    print('Total Securities: {}'.format(len(secs)))
    print('Total no of days: {}'.format(len(df['Date'].unique())))
    
    print(df.isnull().sum())
    
    # Check the statistics of the columns of the merged dataframe and check for outliers
    print(df.describe())

    # plot histogram
    df.hist(sharex = False, sharey = False, xlabelsize = 4, ylabelsize = 4, figsize=(10, 10))
    plt.show()

#    print(df[pd.isnull(df).any(axis=1)])
    
#    if normalize:
#         normalize the data attributes
#        df[''] = preprocessing.normalize(df[''])

    return df


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
    zeros = len(s[s.vol==0.0])
    if zeros>total*empty_thres:
        empty_secs.add(s.sec_id[0])
  
  print("Discarding " + str(len(empty_secs)) + " securities with incomplete data (threshold=" + str(empty_thres))
  clean_series = []
  for s in series:
    if s.sec_id[0] not in empty_secs:
        clean_series.append(s)
  
  #normalize data and fill empty records
  median = org_data['vol'].median()
  input_vars = ['vol','X1','X2','X3','X4','X5','X6','X7']
  clean_series = [ ts_fillna(x, 'vol', median) for x in clean_series] 
  clean_series = [ ts_normalize(x, input_vars) for x in clean_series] 
  
  #make vol a stationary variable using fractional differencing
  #the differencing factor is decided as the value that makes more than 95% 
  #of the samples in the dataset stationary
  [add_vol_ffd(x, 0.7) for x in clean_series]
  
  org_data = pd.concat(clean_series)
    
  #TODO normalize again along the entire set?
  return org_data
    
def run_models_cross_validation(models):
    
    return 0

def get_data():
    df = pd.read_csv('dat_final.csv')
    df.set_index('Date')
#    df=df.loc[df['Date']==0]
    return df

def get(df):
    #X = df[['vol', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']].values
    X = df[['vol_ffd', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']].values
    y = df[['fut_ret']].values.flatten()
    return (X, y)

def get_model(name):
    if name == 'LinearRegression': return LinearRegression()
    elif name == 'DecisionTreeRegressor': return DecisionTreeRegressor()
    elif name == 'RandomForestRegressor': return RandomForestRegressor(n_estimators=50,min_samples_split=0.001)
    elif name == 'KNeighborsRegressor': return KNeighborsRegressor()
    elif name == 'GradientBoostingRegressor': return GradientBoostingRegressor(learning_rate=0.1,n_estimators=20,
                                                                               max_depth=2,max_features=6,
                                                                               min_samples_split=4000,min_samples_leaf=200)
    # cluster based models
    elif name == 'ClusterLinearRegressor': return ClusterRegressor('LinearRegression')
    elif name == 'ClusterGradientBoostingRegressor': return ClusterRegressor('GradientBoostingRegressor')
    else: return LinearRegression()
    
class ClusterRegressor:
    def __init__(self, name):
        self.columns = ['vol', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']
        self.name = name
        
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
        
        df_X = pd.DataFrame(X, columns=self.columns)
        df_y = pd.DataFrame(y, columns=['fut_ret'])
        
        df_X = pd.concat([df_X, clusters], axis=1)
        df_y = pd.concat([df_y, clusters], axis=1)
        
        models = {}
        for cluster in unique_clusters:
            filtered_df_X = df_X.loc[df_X['cluster']==cluster]
            filtered_df_y = df_y.loc[df_y['cluster']==cluster]
            
            X_train = filtered_df_X[self.columns].values
            y_train = filtered_df_y['fut_ret'].values.flatten()
            
            regressor = get_model(self.name)
            print('cluster:{}, model:{}, data size:{}'.format(cluster, self.name, len(y_train)))
            regressor.fit(X_train, y_train)
            models[cluster] = regressor
             
        print('fit complete')
        self.models = models
        
    def predict(self, X_test):
        labels = self.kmeans.predict(X_test)       
        clusters = pd.DataFrame(labels, columns=['cluster'])
        df = pd.DataFrame(X_test, columns=self.columns)
        
        df = pd.concat([df, clusters], axis=1)
        y_pred_arr =[]
        print('start predict')
        for index, row in df.iterrows():
            r = np.reshape(np.array(row[:-1]), (1, len(row)-1))
            y_pred = self.models[row['cluster']].predict(r)
            y_pred_arr.append(y_pred[0])
        
        print('finished predict')
        return y_pred_arr
    
def run():
    df = get_data()
    df = pre_process(df)
    df = clean_data(df)
    df.reset_index(inplace=True)
    # plot for a sec_id = 0
    sec_0 = df.loc[df['sec_id'] == 0]
    sec_0 = sec_0[['Date','fut_ret']]
    sec_0.plot(kind='line',x='Date', y='fut_ret',ax=plt.gca())
    #plt.show()
    
    (X, y) = get(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/6, shuffle=False)
        
    # Append the models to the models list
    model_names = ['LinearRegression', 
                   'KNeighborsRegressor', 
                   'DecisionTreeRegressor', 
                   'RandomForestRegressor',
                   'GradientBoostingRegressor']
    
    # fit
    result = {}
    for name in model_names:
        print('Running model:{}'.format(name))
        cr = get_model(name)
        cr.fit(X_train, y_train)
        r2_out = r2_score(y_test, cr.predict(X_test))
        r2_in = r2_score(y_train, cr.predict(X_train))
        result[name] = (r2_in, r2_out)
        print('{} r2_in : {}, r2_out:{}'.format(name, r2_in, r2_out))
     
    print(result)

run()