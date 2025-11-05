print("Topic: Simple Linear Regression")
 
print() 

print("Step 1: Importing the libraries") 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
import scipy.stats as stats


print("Step 2: Loading the dataset") 

df = pd.read_csv('student_scores.csv') 

print("Step 3: Data preparation: X, y") 

X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values


print("Step 4: Data visualization")


plt.scatter(X, y, color='red', marker='o')   # plot X vs y
plt.xlabel(df.columns[0])                    # label for x‑axis
plt.ylabel(df.columns[1])                    # label for y‑axis
plt.title('Hours vs Score')                  # This will show relationship


print("Step 5: Splitting the dataset") 

X_train, X_test, y_train, y_test = train_test_split( 
X,  
y, 
test_size = 0.2,  
random_state = 0 
) 

print("Step 6: Model creation") 

model = LinearRegression() 

print("Step 7: Model training") 

model.fit(X_train, y_train) 

print("Step 8: Prediction") 

 
y_pred = model.predict(X_test) 
print()

print('y_prediction: ',y_pred)
print()

m = model.coef_              #Coeficient - m

c = model.intercept_         #Intercept - c

print('Coeficient: ',m)
print()
print('Intercept: ',c)
print()

inp = m*X_test + c          #SLR Formula:- y = mx + c
print('prediction: ',inp)

print()
print("Step 9: (R-value, MSE, MAE, RMSE, R-squared)")

# y_test and y_pred already defined

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print()

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²):", r2)
print()
print("Step 10: Calculating Pearson.s r and p-value")
print()

r, p = stats.pearsonr(y_test, y_pred)
print("Pearson correlation coefficient (r):", r)
print()
print("p-value:", p)

plt.plot(X, model.predict(X), color='blue', linewidth=2, label='Regression line')
plt.legend()
plt.show()