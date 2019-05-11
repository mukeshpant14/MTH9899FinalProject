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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score

def pre_process(df, normalize=False):
    secs = df['sec_id'].unique()
    print('Size of dataset:{}'.format(df.shape[0]))
    print('Total Securities: {}'.format(len(secs)))
    print('Total no of days: {}'.format(len(df['Date'].unique())))
    
    print(df.isnull().sum())
#    print(df[pd.isnull(df).any(axis=1)])
    
#    if normalize:
#         normalize the data attributes
#        df[''] = preprocessing.normalize(df[''])

    return df
    
def clean_data(org_data):      
    median = org_data['vol'].median()
    org_data['vol'].fillna(median, inplace=True)

    return org_data
    
def cross_validation():
    return 0

def get_data():
    df = pd.read_csv('dat_final.csv')
    print(df.columns)    
    df.set_index('Date')
    return df

def get(df):
    X = df[['vol', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']].values
    y = df[['fut_ret']].values.flatten()
    return (X, y)

def run():
    df = get_data()
    df = pre_process(df)
    df = clean_data(df)
    
    (X, y) = get(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/6, shuffle=False)

    models = [LinearRegression(),
              RandomForestClassifier(n_estimators=100, max_depth=2,random_state=5)]
            #MLPClassifier(random_state=5)]
    
    # fit
    for model in models:
        model.fit(X_train, y_train)
        
    # predict
    for model in models:
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print("R2 Score: {0:0.5f}".format(r2))    

run()