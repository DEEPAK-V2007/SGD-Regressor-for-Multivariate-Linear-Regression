# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation: Create a synthetic dataset with 3 features (Area, Rooms, Location) and 2 targets (Price, Occupants), then split into train and test sets and apply StandardScaler to normalize the features.
2. Model Building: Wrap SGDRegressor inside MultiOutputRegressor to handle multiple output targets, and configure it with constant learning rate, 1000 iterations and tolerance of 1e-3.
3. Training and Prediction: Fit the MultiOutputRegressor on the scaled training data, then predict both target values (Price and Occupants) simultaneously for the test set.
4. Evaluation: Measure model performance using R2 Score and Mean Squared Error, then compare actual vs predicted values for both output targets to assess accuracy.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: DEEPAK.V
RegisterNumber: 212225230044
*/

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

X = np.array([
    [1200, 3, 1],
    [1500, 4, 2],
    [800, 2, 1],
    [2000, 5, 3],
    [1700, 4, 2],
    [1000, 2, 1],
    [2200, 5, 3],
    [1300, 3, 2]
])

y = np.array([
    [50, 4],
    [65, 5],
    [35, 3],
    [90, 7],
    [70, 6],
    [40, 3],
    [100, 8],
    [55, 4]
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sgd = SGDRegressor(max_iter=5000, tol=1e-3, eta0=0.01, learning_rate='constant')
multi_regressor = MultiOutputRegressor(sgd)
multi_regressor.fit(X_train, y_train)

y_pred = multi_regressor.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("\nActual vs Predicted:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted.round(2)}")
```

## Output:

<img width="457" height="143" alt="{E38C1354-EFF2-442F-88DD-2292FC161DE3}" src="https://github.com/user-attachments/assets/d043e66e-2d67-4aa1-a51e-7354e085dc23" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
