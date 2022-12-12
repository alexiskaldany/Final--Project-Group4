""" 
Use this file to analyze eval_df.csv 
TODO: Sagar
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import numpy as np
from sklearn import metrics
import seaborn as sns
import statsmodels.api as sm

data = pd.read_csv("/Users/sagartripathi/Documents/Final--Project-Group4/visualizations/eval_df.csv")
print(data.head())
target = ['rouge1', 'rouge2', 'rougeL']

############### Regression plot #############################

for var in target:
    plt.figure() # Creating a rectangle (figure) for each plot
    # Regression Plot also by default includes
    # best-fitting regression line
    # which can be turned off via `fit_reg=False`
    sns.regplot(x=data['sum_text_ratio'], y=data[var], data=data,line_kws={"color": "red"}).set(title=f'Regression plot of {var}')
    plt.show()


############# correlation plot ##################

new_dataframe = data.drop(["Unnamed: 0","id","text","summary","mode","pred_summary","loss","epoch","text_length","summary_length"],axis=1)
sns.heatmap(new_dataframe.corr(),annot = True, fmt='.2g',cmap= 'coolwarm')
plt.show()

################ Decision Tree reegressor ##################

# get the locations
X = data[['sum_text_ratio']]
y = data['rouge1']

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, random_state=0)

regressor = DecisionTreeRegressor(random_state=0)

# fit the regressor with X and Y data
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

df=pd.DataFrame({'Actual':y_test, 'Predicted':predictions})

print('Mean Absolute Error for rouge1:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error for rouge1:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error for rouge1:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))



x_ax = range(len(y_test))
plt.plot(x_ax, predictions, linewidth=1, label="original")
plt.plot(x_ax, y_test, linewidth=1.1, label="predicted")
plt.title("y-test and y-predicted data for rouge1")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()



######################### rouge 2 as target ###################

X = data[['sum_text_ratio']]
y = data['rouge2']

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, random_state=0)

regressor = DecisionTreeRegressor(random_state=0)

# fit the regressor with X and Y data
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

df=pd.DataFrame({'Actual':y_test, 'Predicted':predictions})

print('Mean Absolute Error for rouge2:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error for rouge2:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error for rouge2:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))



x_ax = range(len(y_test))
plt.plot(x_ax, predictions, linewidth=1, label="original")
plt.plot(x_ax, y_test, linewidth=1.1, label="predicted")
plt.title("y-test and y-predicted data for rouge2")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()


######################################## rougeL as target #####################

X = data[['sum_text_ratio']]
y = data['rougeL']

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, random_state=0)

regressor = DecisionTreeRegressor(random_state=0)

# fit the regressor with X and Y data
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

df=pd.DataFrame({'Actual':y_test, 'Predicted':predictions})

print('Mean Absolute Error for rougeL:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error for rougeL:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error for rougeL:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


x_ax = range(len(y_test))
plt.plot(x_ax, predictions, linewidth=1, label="original")
plt.plot(x_ax, y_test, linewidth=1.1, label="predicted")
plt.title("y-test and y-predicted data for rougeL")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()

################### end of Decision Tree ######################


################  start of linear regression #####################
##################### rouge1 as target #####################

# get the locations
X = data[['sum_text_ratio']]
y = data[['rouge1']]

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

##################### rouge2 as target #####################

# get the locations
X = data[['sum_text_ratio']]
y = data[['rouge2']]

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


##################### rougeL as target #####################

# get the locations
X = data[['sum_text_ratio']]
y = data[['rougeL']]

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

############### end of linear regression ################


