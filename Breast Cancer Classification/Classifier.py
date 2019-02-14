# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
print(cancer['DESCR'])

# Importing the dataset
#dataset = pd.read_csv('data.csv')
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
df_cancer.head()

sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
sns.countplot(df_cancer['target'])
sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)

sns.heatmap(df_cancer.corr())

X = df_cancer.drop(['target'], axis = 1)
y = df_cancer['target']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

#Fitting Classifier to the Training Set
classifier = SVC()
classifier.fit(X_train, y_train)

#Predicting the Test set results
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot = True)
print(classification_report(y_test, y_pred))

param_grids = {'C': [0.1, 1,10,100], 'gamma':[1,0.1,0.01,0.001], 'kernel':['rbf']}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grids, refit = True, verbose = 4)
grid.fit(X_train, y_train)
grid.best_params_
grid_pred = grid.predict(X_test)

#Making the Confusion Matrix
cm = confusion_matrix(y_test,grid_pred)
sns.heatmap(cm, annot = True)
print(classification_report(y_test, grid_pred))
