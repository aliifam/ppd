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
    filename = 'dt_model.model'
    loaded_model= joblib.load(filename)
   
    df_input = pd.read_csv('input_data.csv')
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

    