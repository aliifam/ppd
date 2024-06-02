# -*- coding: utf-8 -*-
"""
Created on 2020-11-18
@author: Dr. Ganjar Alfian
email : ganjar@dongguk.edu
for teaching purpose only.
"""

import pandas as pd
import joblib
    
if __name__ == '__main__':
    #load the model
    filename = 'dt_model.model'
    loaded_model= joblib.load(filename)
    #create new unlabelled test data
    df_input = pd.DataFrame(columns = ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA',
       'GGT', 'PROT'])
    df_input.loc[0] = [61, 1, 39, 102.9, 27.3, 143.2, 15, 5.38, 4.88, 72.3, 400.3, 73.4]
    #make prediction
    result = loaded_model.predict(df_input)
    #print(result)

    for i in result:
        int_result = int(i)
        if(int_result==0):
            decision='Suspect Blood Donor'
        elif (int_result==1):
            decision='Hepatitis'
        elif (int_result==2):
            decision='Fibrosis'
        elif (int_result==3):
            decision='Cirrhosis'
        else:
            decision='Not defined'
    
        print('Disease is ', decision)

    