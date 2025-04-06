import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
data = {
    'Age': [50, 30, 45, 60, 35, 55],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', None],
    'BMI': [30.1, 28.5, None, 31.2, 27.3, 29.5],
    'Blood Pressure': [120, 80, 130, 110, None, 125],
    'Outcome': [1, 0, 1, 1, 0, None]
}
dataset=pd.DataFrame(data)
print(dataset.isnull().sum())
dataset['BMI']=dataset['BMI'].fillna(dataset['BMI'].mean())
dataset['Gender']=dataset['Gender'].fillna(dataset['Gender'].mode()[0])
dataset['Blood Pressure']=dataset['Blood Pressure'].fillna(dataset['Blood Pressure'].mean())
dataset['Outcome']=dataset['Outcome'].fillna(dataset['Outcome'].mode()[0])
print(dataset)
#here in the dataset gender is a categorical data so we have to convert it into numerical data
dataset['Gender']=dataset['Gender'].map({"Male":0, "Female":1})
print("the dataset after converting the categorical data to numerical is :")
print(dataset) 
#standarding the data so that mean is Zero and variance is one
sk=StandardScaler()
dataset[['Age','BMI','Blood Pressure']]=sk.fit_transform(dataset[['Age','BMI','Blood Pressure']])
print(dataset)
X=dataset[['Age','Gender','BMI','Blood Pressure']]
y=dataset[['Outcome']]
X_test,X_train,y_test,y_train=train_test_split(X,y,test_size=0.2,random_state=42)
print("size of thr x testing data is:")
print(X_test)
print("size of the x training data is :")
print(X_train)