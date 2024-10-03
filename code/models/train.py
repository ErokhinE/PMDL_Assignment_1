import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib
from sklearn.compose import ColumnTransformer

data=pd.read_csv('data\data.csv')
print(data.head())
print(data.info())
print(data.isnull().any())


# le=LabelEncoder()
# for col in data.columns[data.dtypes=='object']:
#     data[col]=le.fit_transform(data[col])
# print(data.info())

data['Gender']=data['Gender'].map({'Male':1,'Female':0,})
data['Interest']=data['Interest'].map({'Unknown':0, 'Sports':1, 'Others':2, 'Technology':3, 'Arts':4})

data['Personality'] = data['Personality'].map({
    'ENFP': 0, 
    'ESFP': 1, 
    'INTP': 2, 
    'INFP': 3, 
    'ENFJ': 4, 
    'ENTP': 5, 
    'ESTP': 6, 
    'ISTP': 7, 
    'INTJ': 8, 
    'INFJ': 9, 
    'ISFP': 10, 
    'ENTJ': 11, 
    'ESFJ': 12, 
    'ISFJ': 13, 
    'ISTJ': 14, 
    'ESTJ': 15
})
# print(data['Personality'].unique())
x=data.drop(columns='Personality')
y=data['Personality']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20)


model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

# Прогнозирование на тестовом наборе
y_pred = model.predict(x_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print("Точность:", accuracy)


joblib.dump(model, 'C:/Users/erzhe/PMDL_Assignment_1/models/random_forest_model.pkl')

