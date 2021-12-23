# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:33:44 2021

@author: Peyman
"""

'''a.1'''
import pandas as pd
# set pandas options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
import os
path = "C:/Users/Peyman/.spyder-py3/Assignment4"
filename = 'titanic.csv'
fullpath = os.path.join(path, filename)
titanic_peyman = pd.read_csv(fullpath)

'''b.1'''
print(titanic_peyman.head(3))

'''b.2'''
print(titanic_peyman.shape)

'''b.3'''
print(titanic_peyman.columns.values)
print(titanic_peyman.dtypes)
titanic_peyman.info()

'''b.4'''
titanic_peyman = titanic_peyman.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

'''b.5'''
print(titanic_peyman['Sex'].unique())
print(titanic_peyman['Pclass'].unique())

'''c.1.a'''
import matplotlib.pyplot as plt
pd.crosstab(titanic_peyman.Survived, titanic_peyman.Pclass)
pd.crosstab(titanic_peyman.Survived, titanic_peyman.Pclass).plot(kind='bar')
plt.title('Survival Frequency for Passenger Class')
plt.xlabel('Survival')
plt.ylabel('Frequency of Passenger Class')

'''c.1.b'''
pd.crosstab(titanic_peyman.Survived, titanic_peyman.Sex)
pd.crosstab(titanic_peyman.Survived, titanic_peyman.Sex).plot(kind='bar')
plt.title('Survival Frequency for Gender')
plt.xlabel('Survival')
plt.ylabel('Frequency of Passenger Class')

'''c.2 d.1-4'''
titanic_peyman = pd.get_dummies(titanic_peyman)
pd.plotting.scatter_matrix(titanic_peyman, figsize = (20, 20))

'''d.5'''
titanic_peyman["Age"].fillna(titanic_peyman["Age"].mean(), inplace = True)

'''d.6'''
titanic_peyman = titanic_peyman.astype(float)

'''d.7'''
titanic_peyman.info()

'''d.8'''
def Normalize(df):
    cols = df.columns
    min = df.min()
    max = df.max()
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            df.at[i, cols[j]] = (df.loc[i][j] - min[j]) / (max[j] - min[j])
    return df

'''d.9'''
titanic_peyman = Normalize(titanic_peyman)

'''d.10'''
print(titanic_peyman.head(2))

'''d.11'''
titanic_peyman.hist(figsize=(9, 10))

'''d.13'''
x_peyman = titanic_peyman[[x for x in titanic_peyman.columns.values if x != 'Survived']]
y_peyman = titanic_peyman["Survived"]

'''d.13.i'''
from sklearn.model_selection import train_test_split
x_train_peyman, x_test_peyman, y_train_peyman, y_test_peyman = train_test_split(x_peyman, y_peyman, test_size=0.3, random_state=8)

'''e.1'''
from sklearn.linear_model import LogisticRegression
peyman_model = LogisticRegression()
peyman_model.fit(x_train_peyman, y_train_peyman)

'''e.2'''
import numpy as np
pd.DataFrame(zip(x_train_peyman.columns, np.transpose(peyman_model.coef_)))

'''e.3.1-2'''
from sklearn.model_selection import cross_val_score
scores = cross_val_score(peyman_model, x_train_peyman , y_train_peyman , scoring='accuracy', cv=10)
print (scores)
print (scores.mean())

'''e.3.3-5'''
print("min        ", "max        ", "mean")
for i in np.arange (0.10, 0.5, 0.05):
    x_train_peyman, x_test_peyman, y_train_peyman, y_test_peyman = train_test_split(x_peyman, y_peyman, test_size=i, random_state=8)
    peyman_model.fit(x_train_peyman, y_train_peyman)
    scores = cross_val_score(peyman_model, x_train_peyman, y_train_peyman, scoring='accuracy', cv=10)
    print("{:.9f} {:.9f} {:.9f}".format(scores.min(), scores.max(), scores.mean()))

'''b.1'''
x_train_peyman, x_test_peyman, y_train_peyman, y_test_peyman = train_test_split(x_peyman, y_peyman, test_size=0.3, random_state=8)
peyman_model.fit(x_train_peyman, y_train_peyman)

'''b.2'''
y_pred_peyman = peyman_model.predict_proba(x_test_peyman)

'''b.3'''
y_pred_peyman_flag = y_pred_peyman[:,1] > 0.5

'''b.4'''
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

'''b.5'''
print(accuracy_score(y_test_peyman, y_pred_peyman_flag))

'''b.6'''
print(confusion_matrix(y_test_peyman, y_pred_peyman_flag))

'''b.7'''
print(classification_report(y_test_peyman, y_pred_peyman_flag))

'''b.9'''
y_pred_peyman_flag = y_pred_peyman[:,1] > 0.75
print(accuracy_score(y_test_peyman, y_pred_peyman_flag))
print(confusion_matrix(y_test_peyman, y_pred_peyman_flag))
print(classification_report(y_test_peyman, y_pred_peyman_flag))

'''b.10'''
y_pred_peyman = peyman_model.predict_proba(x_train_peyman)
y_pred_peyman_flag = y_pred_peyman[:,1] > 0.75
print(accuracy_score(y_train_peyman, y_pred_peyman_flag))
print(confusion_matrix(y_train_peyman, y_pred_peyman_flag))
print(classification_report(y_train_peyman, y_pred_peyman_flag))