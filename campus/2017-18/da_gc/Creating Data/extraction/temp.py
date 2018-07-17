# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Have to make changes over here considering the data used
# Importing the dataset
dataset = pd.read_csv('Exported.csv')
X = dataset.iloc[:, [0,1,2,3,4,5,6]].values
y = dataset.iloc[:, 7].values

a = pd.isnull(dataset['AUM'])
ids = []
for i in range(len(a)):
    if a[i]:
        ids.append(i)
        
d = dataset.drop(dataset.index[ids])

x = d['AUM'].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x.reshape(-1,1))
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x.reshape(-1,1))

#Now we have predicted classes of dataset
temp = dataset.dropna(thresh = 50000, axis = 1)
#Encoding the data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from fancyimpute import KNN
data_1 = KNN(k = 3).complete(temp)
#labelencoder_X = LabelEncoder()
#X[:, [4,5,6]] = labelencoder_X.fit_transform(X[:, [4,5,6])
onehotencoder = OneHotEncoder(categorical_features = [4,5,6])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_kmeans, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train_sc = sc_X.fit_transform(X_train)
X_test_sc = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train_sc = sc_y.fit_transform(y_train)

#UsingSVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train_sc, y_train_sc)

# Predicting the Test set results
y_pred1 = classifier.predict(X_test_sc)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)
importance1 = classifier.coef_

#Using Kernel-SVM
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train_sc, y_train_sc)

# Predicting the Test set results
y_pred2 = classifier.predict(X_test_sc)

# Making the Confusion Matrix
cm2 = confusion_matrix(y_test, y_pred2)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train_sc, y_train_sc)

# Predicting the Test set results
y_pred3 = classifier.predict(X_test_sc)

# Making the Confusion Matrix
cm3 = confusion_matrix(y_test, y_pred3)
importance2 = classifier.feature_importances_

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred4 = classifier.predict(X_test)

# Making the Confusion Matrix
cm4 = confusion_matrix(y_test, y_pred4)
importance3 = classifier.feature_importances_

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Fitting LightGBM to the Training set
import lightgbm as lgb
gbm = lgb.LGBMClassifier(n_estimators=2900, max_depth=3, subsample=0.7, colsample_bytree= 0.7)
gbm = gbm.fit(X_train, y_train)
importance4 = gbm.feature_importance()

# Predicting the Test set results
to_use = gbm.predict_proba(X_test)

y_pred5=[]
temp = 0
import operator
for i in range(len(to_use)) :
    temp = to_use.ix[i]
    max_index, max_value = max(enumerate(temp), key=operator.itemgetter(1))
    y_pred5.append(max_index)
    
cm5 = confusion_matrix(y_test, y_pred5)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_sc, y_train_sc)

# Predicting the Test set results
y_pred6 = classifier.predict(X_test_sc)

# Making the Confusion Matrix
cm6 = confusion_matrix(y_test, y_pred6)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train_sc, y_train_sc, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred7 = classifier.predict(X_test_sc)
y_pred7 = (y_pred7 > 0.5)

# Making the Confusion Matrix
cm7 = confusion_matrix(y_test, y_pred7)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train_sc, y_train_sc)

# Predicting the Test set results
y_pred8 = classifier.predict(X_test_sc)

# Making the Confusion Matrix
cm8 = confusion_matrix(y_test, y_pred8)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train_sc, y_train_sc)

# Predicting the Test set results
y_pred9 = classifier.predict(X_test_sc)

# Making the Confusion Matrix
cm9 = confusion_matrix(y_test, y_pred9)

# ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=200, random_state=23)
forest.fit(X_train,y_train)
importance5 = forest.feature_importances_

y_pred10 = forest.predict(X_test)
cm10 = confusion_matrix(y_test, y_pred10)

from pd import DataFrame
PredFrame = DataFrame({
        'SVM':y_pred1,
        'KSVM':y_pred2,
        'RFC':y_pred3,
        'XGB':y_pred4,
        'LGBM':y_pred5,
        'LR':y_pred6,
        'ANN':y_pred7,
        'KNN':y_pred8,
        'DTC':y_pred9,
        'ETC':y_pred10, 
         })
    
classifier = XGBClassifier()
classifier.fit(PredFrame, y_test)

important_importance = classifier.feature_importances_

w_svm = important_importance[0]
w_rf = important_importance[2]
w_xgb = important_importance[3]
w_lgbm = important_importance[4]
w_etc = important_importance[9]

w_sum = w_svm + w_rf + w_xgb + w_lgbm + w_etc

final_importance = w_svm*importance1 + w_rf*importance2 + w_xgb*importance3 + w_lgbm*importance4 + w_etc*importance5
final_importance = final_importance / w_sum