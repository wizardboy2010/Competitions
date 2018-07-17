

import numpy as np
import pandas as pd

data = pd.read_csv('train.csv')
data1 = data.drop('PassengerId',1)
data1 = data1.drop('Name',1)
data1 = data1.drop('Embarked',1)
data1 = data1.drop('Cabin',1)
data1 = data1.drop('Ticket',1)
data1 = data1.drop('Pclass',1)
data1 = data1.drop('Fare',1)

x = data1.iloc[:,1:].values
y = data1.iloc[:,0].values

from sklearn.preprocessing import LabelEncoder, StandardScaler, Imputer, OneHotEncoder
imputer = Imputer(strategy = "mean")
imputer = imputer.fit(x[:,1].reshape(-1,1))
x[:,1] = np.reshape(imputer.transform(x[:,1].reshape(-1,1)),(891,))
le = LabelEncoder()
x[:,0] = le.fit_transform(x[:,0])
sc = StandardScaler()
x[:,1:] = sc.fit_transform(x[:,1:])
onehotencoder = OneHotEncoder()
y = onehotencoder.fit_transform(y.reshape(-1,1)).toarray()