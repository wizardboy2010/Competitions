import pandas as pd
import numpy as np
import os

try:
    path = os.path.dirname(os.path.realpath(__file__))
except:
    path = 'E:/ubuntu_data/Projects/Competitions/campus/2018-19/Analyse_This/src'

os.chdir(path)

data_path = os.path.join(os.path.dirname(path), 'data')

data = pd.read_csv(os.path.join(data_path,'Training_dataset_Original.csv'), low_memory=False)

def assign_missing(val):
    if val in ['missing', 'na', 'NA', 'Na', 'Nan', 'NaN', 'N/A']:
        return np.NaN
    try:
        return float(val)
    except:
        return str(val)

data = data.applymap(lambda x: assign_missing(x))

################## For charge card

c_data = data[data.mvar47=='C']

c_data = c_data.drop('mvar47', 1)

rem = [i for i in c_data.columns if sum(pd.isna(c_data[i])) >= 0.2*len(c_data)]

for i in rem:
    c_data = c_data.drop(i,1)

des = c_data.describe()

for i in des.columns:
    if des[i]['count'] != len(c_data):
        c_data[i] = c_data[i].fillna(des[i]['mean'])

x = c_data.values[:,1:-1]
y = c_data.values[:,-1].astype(int)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x[:,:] = sc.fit_transform(x[:,:])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2 ,random_state = 42)

###### 9th trail
from sklearn.ensemble import RandomForestClassifier

for i in range(5, 25):
    print('for', i)
    rn_2 = RandomForestClassifier(n_estimators=i, min_samples_split=13, random_state=42)
    rn_2.fit(x_train, y_train)
    print('train acc', rn_2.score(x_train, y_train))
    print('test acc', rn_2.score(x_test, y_test),'\n')

rn_2 = RandomForestClassifier(n_estimators=20, min_samples_split=13, random_state=42)
rn_2.fit(x_train,y_train)
print('train acc', rn_2.score(x_train, y_train))
print('test acc', rn_2.score(x_test, y_test))

