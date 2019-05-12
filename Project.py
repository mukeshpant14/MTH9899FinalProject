# -*- coding: utf-8 -*-
"""
Created on Sat May 11 20:28:39 2019

@author: ravee
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score

def pre_process(df, normalize=False):
    secs = df['sec_id'].unique()
    print('Size of dataset:{}'.format(df.shape[0]))
    print('Total Securities: {}'.format(len(secs)))
    print('Total no of days: {}'.format(len(df['Date'].unique())))
    
    print(df.isnull().sum())
#    print(df[pd.isnull(df).any(axis=1)])
    df=clean_data(df)
#    if normalize:
    cols=['fut_ret','vol','X1','X2','X3','X4','X5','X6','X7']
    for col in cols:
        col_zcore=col+'_norm'
        df[col_zcore]=(df[col] - df[col].mean())/df[col].std(ddof=0)
        df.describe()
    return df
    
def clean_data(org_data):      
    median = org_data['vol'].median()
    org_data['vol'].fillna(median, inplace=True)

    return org_data
    
def cross_validation():
    return 0

def get_data():
    df=pd.read_csv('C:\MTH 9899 Machine Learning 2\dat_final.csv')
    df.head()
    print(df.columns)    
    df.set_index('Date')
    return df

def featureselection(df):
    model=LinearRegression()
    rfe = RFE(model, 6)
    (X, y) = get(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/6, shuffle=False)
    rfe = rfe.fit(X_train, y_train)
    names=['vol_norm', 'X1_norm', 'X2_norm', 'X3_norm', 'X4_norm', 'X5_norm', 'X6_norm','X7_norm']

    print ("Features by RFE process:")
    print (sorted(zip(map(lambda x: x, rfe.support_), 
                  names), reverse=True))
    print(rfe.support_)
    print(rfe.ranking_)

    

def get(df):
    X = df[['vol_norm', 'X1_norm', 'X2_norm', 'X3_norm', 'X4_norm', 'X5_norm', 'X6_norm','X7_norm']].values
    y = df[['fut_ret_norm']].values.flatten()
    return (X, y)


def gbm(df):
    
    estimator=GradientBoostingRegressor()
    estimator.fit()
    
def run():
    df = get_data()
    df = pre_process(df,True)
    df = clean_data(df)
    print(df.columns)
    featureselection(df)
    (X, y) = get(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/6, shuffle=False)
    
    #models = [LinearRegression(),
              #RandomForestClassifier(n_estimators=100, max_depth=2,random_state=5)]
            #MLPClassifier(random_state=5)]
    
    # fit
    #for model in models:
        #model.fit(X_train, y_train)
        
    # predict
    #for model in models:
        #y_pred = model.predict(X_test)
        #r2 = r2_score(y_test, y_pred)
        #print("R2 Score: {0:0.5f}".format(r2))    

run()