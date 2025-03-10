# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NAVEEN KUMAR S
RegisterNumber: 212223040129
*/
```
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

x.shape
y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
x_train.shape
x_test.shape
y_train.shape
y_test.shape

reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```

## Output:
### Head values:
![image](https://github.com/user-attachments/assets/4b0410bd-8c39-4893-8cb5-a8b2e05dfebb)

### Tail values:
![image](https://github.com/user-attachments/assets/8962fd15-5563-41f9-9b63-2d29649868a0)

### Dataset info:
![image](https://github.com/user-attachments/assets/8046d6a2-8e01-44bb-b2d2-fbba326508a9)

### Compare Dataset:
![image](https://github.com/user-attachments/assets/78a70fd7-c860-41c9-817c-aea4ae038e89)

### Y_prediction values:
![image](https://github.com/user-attachments/assets/62ec82af-c1c1-49ef-b304-395e015f05aa)

### Training set:
![image](https://github.com/user-attachments/assets/b4842eec-52be-4dbd-9480-84a049327451)

### Testing set:
![image](https://github.com/user-attachments/assets/903a1336-f5fb-4dad-ad90-7c59c80b37c3)

### MSE,MAE,RMSE:
![image](https://github.com/user-attachments/assets/d5d3501e-6219-4654-a510-3cafba26ce38)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
