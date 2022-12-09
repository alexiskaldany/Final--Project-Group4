""" 
Use this file to analyze eval_df.csv 
TODO: Sagar
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import numpy as np
from sklearn import metrics
import seaborn as sns

data = pd.read_csv("/Users/sagartripathi/Documents/Final--Project-Group4/visualizations/eval_df.csv")
print(data.head())
features = ['rouge1', 'rouge2', 'rougeL']
target = 'sum_text_ratio'

############### Regression plot #############################

for var in features:
    plt.figure() # Creating a rectangle (figure) for each plot
    # Regression Plot also by default includes
    # best-fitting regression line
    # which can be turned off via `fit_reg=False`
    sns.regplot(x=data[var], y=data['sum_text_ratio'], data=data,line_kws={"color": "red"}).set(title=f'Regression plot of {var} and Petrol Consumption')
    plt.show()


############# correlation plot ##################

new_dataframe = data.drop(["Unnamed: 0","id","text","summary","mode","pred_summary","loss","epoch","text_length","summary_length"],axis=1)
sns.heatmap(new_dataframe.corr(),annot = True, fmt='.2g',cmap= 'coolwarm')
plt.show()

################ Decision Tree reegressor ##################

# get the locations
X = data.drop(["Unnamed: 0","id","text","summary","mode","pred_summary","loss","epoch","text_length","summary_length","sum_text_ratio"],axis=1)
y = data[target]

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, random_state=0)

regressor = DecisionTreeRegressor(random_state=0)

# fit the regressor with X and Y data
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

df=pd.DataFrame({'Actual':y_test, 'Predicted':predictions})

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

plt.figure(figsize=(20,20), dpi=200)
plot_tree(regressor, feature_names=X.columns)
plt.show()




################ linear regression #####################


features = ['rouge1', 'rouge2', 'rougeL']
target = 'sum_text_ratio'

# get the locations
X = data.drop(["Unnamed: 0","id","text","summary","mode","pred_summary","loss","epoch","text_length","summary_length","sum_text_ratio"],axis=1)
y = data[target]

ols = linear_model.LinearRegression()
model = ols.fit(X, y)
print('Features                :  %s' % features)
print('Regression Coefficients : ', [round(item, 2) for item in model.coef_])
print('R-squared               :  %.2f' % model.score(X, y))
print('Y-intercept             :  %.2f' % model.intercept_)
print('')


