# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start Step
2. Data Preparation 
3. Hypothesis Definition
4. Cost Function
5. Parameter Update Rule
6. Iterative Training
7. Model Evaluation
8. End

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: keerthana S
RegisterNumber: 212223040092 
*/

```
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


```
```
data = fetch_california_housing()
```
```
print(data)
```
## Output:
![image](https://github.com/user-attachments/assets/91d870d3-c4ce-4b1d-bcd4-4ea4002982bb)
```
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
```
## Output:
![image](https://github.com/user-attachments/assets/e6ed3499-aacf-41e9-890b-b8805331929f)
```
df.info()

```
## Output:
![image](https://github.com/user-attachments/assets/109efd14-480b-4d57-8e5d-6719da4821db)
```
X=df.drop(columns=['AveOccup','target'])

```
```
X.info

```
## Output:
![image](https://github.com/user-attachments/assets/d40daf0b-9173-4aae-bd07-0c5b54228d99)
```
Y=df[['AveOccup','target']]
```
```
print(Y.info())
```
## Output:
![image](https://github.com/user-attachments/assets/dc1363db-1857-4710-9156-877e20ddac12)
```
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
```
```
X.head()
```
## Output:
![image](https://github.com/user-attachments/assets/ea2ea688-0ecc-4875-a9da-ee3541ad4d12)
```
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.fit_transform(Y_test)
```
```
print(X_train)
```
## Output:
![image](https://github.com/user-attachments/assets/d2d8213c-aebd-4329-a375-7dc898949895)
```
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
```
```
multi_output_sgd = MultiOutputRegressor(sgd)

```
```
multi_output_sgd.fit(X_train, Y_train)
```
# Output:
![image](https://github.com/user-attachments/assets/9f973489-65cd-4e32-8f78-53ad9d107f5f)

```
Y_pred = multi_output_sgd.predict(X_test)
```
```
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
```
```
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
```
## Output:
![image](https://github.com/user-attachments/assets/7fe7ab85-ea54-462b-9cff-2fde37e432f7)
```
print("\nPredictions:\n", Y_pred[:5])
```
## Output:
![image](https://github.com/user-attachments/assets/ea4f0bd2-91cb-40fe-b4ee-5d1a69f9a2af)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
