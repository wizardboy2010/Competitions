import pandas as pd
import os
import numpy as np

# path = os.path.dirname(os.path.realpath(__file__))
# print(path)
# data_path = os.path.join(os.path.dirname(path), 'data')
# data = pd.read_csv(os.path.join(data_path,'Training_dataset_Original.csv'))

data = pd.read_csv('campus/2018-19/Analyse_This/data/Training_dataset_Original.csv', low_memory=False)

def assign_missing(val):
    if val in ['missing', 'na', 'NA', 'Na', 'Nan', 'NaN', 'N/A']:
        return np.NaN
    try:
        return float(val)
    except:
        return str(val)

data = data.applymap(lambda x: assign_missing(x))

#######  first and second
rem = [i for i in data.columns if sum(pd.isna(data[i])) >= 10000]

###### third
rem = [i for i in data.columns if sum(pd.isna(data[i])) >= 30000]

for i in rem:
    data = data.drop(i,1)

des = data.describe()

for i in des.columns:
    if des[i]['count'] != len(data):
        data[i] = data[i].fillna(des[i]['mean'])

x = data.values[:,1:-1]
y = data.values[:,-1].astype(int)

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

sc = StandardScaler()
x[:,:-1] = sc.fit_transform(x[:,:-1])
x[:,-1] = [1 if i =='C' else 0 for i in x[:,-1]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2,random_state = 42)

from sklearn.ensemble import RandomForestClassifier

####################### second try
for i in range(2, 15):
    print('for', i)
    first_check_model = RandomForestClassifier(n_estimators=23, min_samples_split=i, random_state=42)
    first_check_model.fit(x_train, y_train)
    print('train acc', first_check_model.score(x_train, y_train))
    print('test acc', first_check_model.score(x_test, y_test),'\n')

first_check_model = RandomForestClassifier(n_estimators=23, min_samples_split=8, random_state=42)
first_check_model.fit(x_train, y_train)
print('train acc', first_check_model.score(x_train, y_train))
print('test acc', first_check_model.score(x_test, y_test),'\n')

####################### third try
for i in range(22, 30):
    print('for', i)
    first_check_model = RandomForestClassifier(n_estimators=i, min_samples_split=19, random_state=42)
    first_check_model.fit(x_train, y_train)
    print('train acc', first_check_model.score(x_train, y_train))
    print('test acc', first_check_model.score(x_test, y_test),'\n')

first_check_model = RandomForestClassifier(n_estimators=27, min_samples_split=19, random_state=42)
first_check_model.fit(x_train, y_train)
print('train acc', first_check_model.score(x_train, y_train))
print('test acc', first_check_model.score(x_test, y_test),'\n')

#################### fourth try

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score

def model_1(input_size):
    model = Sequential()
    model.add(Dense(30, input_dim=input_size, activation = 'relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(10, input_dim=input_size, activation = 'relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))

    adam = Adam(lr = 0.001)

    model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

nn_1 = model_1(x.shape[1])

nn_1.fit(x_train, y_train, 
        epochs = 20, 
        batch_size = 32, 
        validation_data=(x_test, y_test))

print('train_acc', accuracy_score(y_train,nn_1.predict_classes(x_train)))
print('test_acc', accuracy_score(y_test,nn_1.predict_classes(x_test)))

#########################################################################################
## Submission

# data = pd.read_csv(os.path.join(data_path,'Evaluation_dataset.csv.csv'))

sub_data = pd.read_csv('campus/2018-19/Analyse_This/data/Leaderboard_dataset.csv', low_memory=False)

sub_data = sub_data.applymap(lambda x: assign_missing(x))

for i in rem:
    sub_data = sub_data.drop(i,1)

sub_des = sub_data.describe()

for i in sub_des.columns:
    if sub_des[i]['count'] != len(sub_data):
        sub_data[i] = sub_data[i].fillna(sub_des[i]['mean'])

sub_x = sub_data.values[:,1:]
sub_x[:,:-1] = sc.transform(sub_x[:,:-1])
sub_x[:,-1] = [1 if i =='C' else 0 for i in sub_x[:,-1]]

key = sub_data['application_key'].astype(int)
sub = pd.DataFrame(index = key, columns=['default'])

########## first try
sub.default = first_check_model.predict(sub_x)
sub.to_csv('Loosers_IITKgp_2.csv', header=False)

########## second try
sub.default = first_check_model.predict(sub_x)
sub.to_csv('Loosers_IITKgp_3.csv', header=False)

########## third try
sub.default = first_check_model.predict(sub_x)
sub.to_csv('Loosers_IITKgp_4.csv', header=False)

########## fourth try
sub.default = nn_1.predict_classes(sub_x)
sub.to_csv('Loosers_IITKgp_5.csv', header=False)

