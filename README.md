# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program
2. Data Preparation 
3. Hypothesis Definition
4. Cost Function
5. Parameter Update Rule
6. Iterative Training
7. Model Evaluation
8. Stop the program.

## Program:

Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.

Developed by: SATHYAA R

RegisterNumber: 212223100052

```
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

#load the california housing dataset
data=fetch_california_housing()

#use the first 3 feature as inputs
X=data.data[:,:3] #features: 'MedInc' , 'HouseAge' , 'AveRooms'

#use 'MedHouseVal' and 'AveOccup' as output variables
Y=np.column_stack((data.target, data.data[:,6])) #targets: 'MedHouseVal' , 'AveOccup'

#split the data into training and testing sets 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

#scale the features and target variables
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

#initialize the SGDRegressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3)

#use multioutputregressor to handle multiple output variables
multi_output_sgd = MultiOutputRegressor(sgd)

#train the model
multi_output_sgd.fit(X_train,Y_train)

#predict on the test data
Y_pred = multi_output_sgd.predict(X_test)

#inverse transform the predictions to get them back to the original scale
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)

#evaluate the model using mean squared error
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

#optionally, print some predictions
print("\nPredictions:\n",Y_pred[:5]) #print first 5 predictions

```

## Output:


![Screenshot 2024-09-04 203848](https://github.com/user-attachments/assets/28ec2488-36f0-4838-9cf2-258ae37079be)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
