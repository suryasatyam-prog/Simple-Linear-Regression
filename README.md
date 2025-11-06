# Simple Linear Regression: Hours → Score

**Dataset:** `student_scores.csv`  
(Predicting a student’s score based on hours studied)

---

## Table of Contents  
1. [Import Libraries & Load Data](#1-import-libraries--load-data)  
2. [Explore Data & Visualise Relationship](#2-explore-data--visualise-relationship)  
3. [Choose the Machine-Learning Model](#3-choose-the-machine-learning-model)  
4. [Prepare Data, Split & Implement Model](#4-prepare-data,-split-&-implement-model)  
5. [Extract Parameters & Make Predictions](#5-extract-parameters-&-make-predictions)  
6. [Evaluate Model & Visualise Results](#6-evaluate-model-&-visualise-results)  
7. [Interpret Results & Reflect](#7-interpret-results-&-reflect)  
8. [Summary](#8-summary)  

---

## 1. Import Libraries & Load Data  
I started by importing the necessary libraries to perform data analysis and modelling:
 `pandas`, `numpy`, `matplotlib`, `seaborn`, plus modules from `scikit-learn` 
 (for modelling) and `scipy.stats` (for correlation).  
Then I loaded the dataset `student_scores.csv` into a DataFrame using pandas.

---

## 2. Explore Data & Visualise Relationship  
Using pandas and numpy methods, I inspected the dataset: checked its shape, columns, 
basic statistics, and looked out for any missing or invalid values.  
Then I used data visualisation to get a feel for the relationship between the 
independent variable (hours studied) and the dependent variable (score). Specifically,
 I used a scatter plot (with `matplotlib` or `seaborn`) **and drew a line through the
 data** to help see whether the relationship looked roughly linear. This visual check
 gave me an idea that a linear model could be appropriate.

> Example plot:  
> ![Scatter plot with regression line][("plotting\Figure_1.png") ](https://github.com/suryasatyam-prog/Simple-Linear-Regression/blob/main/Figure_1.png) 

---

## 3. Choose the Machine-Learning Model  
Since my visualisation suggested a roughly straight-line relationship, and because I had
 one predictor (hours) and one outcome (score), I chose to use the simple linear regression 
 algorithm.  
I reviewed the backend maths: the model can be expressed as  
\[
\hat y = B_0 + B_1\,x
\]  
where \(B_0\) is the intercept and \(B_1\) is the coefficient (slope).  
This equation guided my understanding of how the model will approximate the relationship
 between study hours and score.

---

## 4. Prepare Data, Split & Implement Model  
- I defined my predictor `X` (hours studied) and target `y` (score).  
- I **split the data into four parts**: training features, test features, training target,
 and test target (using `train_test_split` from `scikit-learn`).  
- I imported `LinearRegression` from the `linear_model` sub-module of `scikit-learn` and 
created a model object.  
- I trained (fitted) the model using the `.fit()` method on the training data.  
- I used the `.predict()` method to predict scores for the test features.

---

## 5. Extract Parameters & Make Predictions  
After the model was trained, I retrieved:  
- The coefficient \(m\) (slope) = the value of `model.coef_`.  
- The intercept \(c\) = the value of `model.intercept_`.  
These values correspond to the equation  
\[
\hat y = m \cdot x + c
\]  
Then I used this formula manually on the test set to check predictions:  
\[
\hat y = m \cdot x + c
\]

---

## 6. Evaluate Model & Visualise Results  
I evaluated how well the model performed on the test set using:  
- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- Coefficient of Determination (R²)  
I also calculated the Pearson correlation coefficient \(r\) and its p-value between actual
 and predicted scores to assess the strength of their linear relationship.  
Finally, I plotted the regression line over the original scatter plot (all data) to visually
 inspect how well the line fits the data.

> Example full-data plot:  
> ![Regression line over data]("plotting\Figure_2.png")  

---

## 7. Interpret Results & Reflect  
From the results, I interpreted the meaning of \(B_1\) (the slope) — for example: “for each
 additional hour of study, the model predicts an increase of about \(B_1\) points in the score.”  
The intercept \(B_0\) gives the predicted score when study hours = 0 (though interpreting
 that in context is important).  
I also reflected on whether the model assumptions (linearity, independence of errors, constant
 error variance, etc.) seemed reasonably satisfied given the data and plots.

---

## 8. Summary  
In this project I used the `student_scores.csv` dataset, visualised the data to confirm a roughly
 linear relationship, decided on simple linear regression, split the data into training & test 
 sets, trained the model end-to-end, extracted the parameters, made predictions, evaluated 
 performance, and finally interpreted the results in context.  
By doing this I gained hands-on experience with the full workflow of a simple regression problem 
— from data exploration all the way to deployment-ready understanding.
