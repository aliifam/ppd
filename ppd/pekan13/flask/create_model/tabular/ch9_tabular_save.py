# -*- coding: utf-8 -*-
"""
Created on 2020-11-18
@author: Dr. Ganjar Alfian
email : ganjar@dongguk.edu
for teaching purpose only.
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import joblib

def dt(X, y):
    model = DecisionTreeClassifier(random_state=0)
    model = model.fit(X, y)
    filename = 'dt_model.model'
    joblib.dump(model, filename)
    
if __name__ == '__main__':
    #read the dataset
    pd.options.display.max_columns=None
    df_ori = pd.read_csv('hcvdat0.csv')
    df_ori = df_ori.dropna()
    #get the X and y
    df_X = df_ori.drop(['Unnamed: 0','Category'],axis=1)
    df_y = df_ori['Category']
    #relabelling the class
    df_y=df_y.replace('0=Blood Donor',0)
    df_y=df_y.replace('0s=suspect Blood Donor',0)
    df_y=df_y.replace('1=Hepatitis',1)
    df_y=df_y.replace('2=Fibrosis',2)
    df_y=df_y.replace('3=Cirrhosis',3)
    
    #get the numerical attributes
    df_X_numb = df_X.drop(['Sex'], axis=1)
    #convert single categorical column into numeric, label encoding    
    le = preprocessing.LabelEncoder()
    df_X['Sex'] = le.fit_transform(df_X['Sex'])
    
    #combine numerical attributes with a newly converted attribute (categorical to numeric, Sex column)
    df_X_new = pd.concat([df_X['Sex'], df_X_numb], axis=1)
    #generate the model 
    print('Generate Decision Tree Model')
    dt(df_X_new, df_y)
    

    