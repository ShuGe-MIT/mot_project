import numpy as np
import pandas as pd
import pickle

file_name = 'Original Epitaxial Layer Growth long V2.csv'
value_col = 'value' # change this to the column name of the value for the first dimension
margin_col = ['V1', 'V2'] # change this to the list of columns to be used as margins
data = pd.read_csv('data/' + file_name).dropna(subset=[value_col]).drop(columns = ["Unnamed: 0"])
value = data.groupby(margin_col)[value_col].apply(list)
tmp = value.values

with open('data/multifactorEpitaxialgrowth.pkl', 'wb') as f:
    pickle.dump([[np.array(tmp[0]), 
                  -np.array(tmp[1]), 
                  -np.array(tmp[2]),
                  np.array(tmp[3])]], f)

with open('data/finitesampleEpitaxialgrowth.pkl', 'wb') as f:
    pickle.dump([[-np.array(tmp[1]), 
                  0.5 * np.array(tmp[2]),
                  0.5 * np.array(tmp[3])]], f)

# import csv file
file_name = 'educationCovariance.csv' # change this to the name of the file
value_col = 'GPA_year1' # change this to the column name of the value for the first dimension
value_col1 = 'GPA_year2' # change this to the column name of the value for the second dimension
margin_col = ['sfp', 'ssp'] # change this to the list of columns to be used as margins

data = pd.read_csv('data/' + file_name).dropna(subset=[value_col, value_col1]).drop(columns = ["Unnamed: 0"])

value = data.groupby(margin_col)[value_col].apply(list)
value1 = data.groupby(margin_col)[value_col1].apply(list)
tmp = value.values
tmp1 = value1.values

with open('data/multidimEducationCovariance.pkl', 'wb') as f:
    pickle.dump([[np.array(tmp[0])/2, 
                  np.array(tmp[1])/2, 
                  -np.array(tmp[2])], 
                 [np.array(tmp1[0])/2, 
                  np.array(tmp1[1])/2, 
                  -np.array(tmp1[2])]], f)

with open('data/multidimInvEducationCovariance.pkl', 'wb') as f:
    pickle.dump([[np.array(tmp[0])/2, 
                  np.array(tmp[1])/2, 
                  -np.array(tmp[2])], 
                 [-np.array(tmp1[0])/2, 
                  -np.array(tmp1[1])/2, 
                  np.array(tmp1[2])]], f)

# import csv file
file_name = 'sample-covariate-simulate.csv' # change this to the name of the file
value_col = 'helpfulness' # change this to the column name of the value for the first dimension
value_col1 = 'altruism' # change this to the column name of the value for the second dimension
margin_col = ['MR', 'experience.variation'] # change this to the list of columns to be used as margins

data = pd.read_csv('data/' + file_name).dropna(subset=[value_col, value_col1]).drop(columns = ["Unnamed: 0"])

value = data.groupby(margin_col)[value_col].apply(list)
value1 = data.groupby(margin_col)[value_col1].apply(list)
tmp = value.values
tmp1 = value1.values

with open('data/multidimHelpfulness.pkl', 'wb') as f:
    pickle.dump([[np.array(tmp[0])/2, 
                  np.array(tmp[1])/2, 
                  -np.array(tmp[2])], 
                 [np.array(tmp1[0])/2, 
                  np.array(tmp1[1])/2, 
                  -np.array(tmp1[2])]], f)
    
from itertools import combinations
file_name = 'educationCovarianceFull.csv'
value_cols = ['GPA_year1', 'GPA_year2', 'grade_20059_fall', 'goodstanding_year1', 'goodstanding_year2']
margin_col = ['sfp', 'ssp']

for value_col, value_col1 in combinations(value_cols, 2):
    data = pd.read_csv('data/' + file_name).dropna(subset=[value_col, value_col1]).drop(columns = ["Unnamed: 0"])
    value = data.groupby(margin_col)[value_col].apply(list)
    value1 = data.groupby(margin_col)[value_col1].apply(list)
    tmp = value.values
    tmp1 = value1.values
    with open('data/multidimEducationCovariance-{}-{}.pkl'.format(value_col, value_col1), 'wb') as f:
        pickle.dump([[np.array(tmp[0])/2, 
                  np.array(tmp[1])/2, 
                  -np.array(tmp[2])], 
                 [np.array(tmp1[0])/2, 
                  np.array(tmp1[1])/2, 
                  -np.array(tmp1[2])]], f)
    with open('data/multidimInvEducationCovariance-{}-{}.pkl'.format(value_col, value_col1), 'wb') as f:
        pickle.dump([[np.array(tmp[0])/2, 
                  np.array(tmp[1])/2, 
                  -np.array(tmp[2])], 
                 [-np.array(tmp1[0])/2, 
                  -np.array(tmp1[1])/2, 
                  np.array(tmp1[2])]], f)