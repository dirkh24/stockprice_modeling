# imports
import pandas as pd
import pandas_datareader.data as web
from pandas import Series, DataFrame

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
#import matplotlib as mpl

import os
import datetime
import math

from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline

# imports for the linear models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

# Adjusting the style of matplotlib
style.use('ggplot')

# fetch the data from the internet
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2019, 9, 7)

df = web.DataReader("AAPL", 'yahoo', start, end)
print(df.tail())

# Plot the close price for the whole timespan
#plt.plot('Adj Close', data=df)
#plt.title('Plot the whole data from yahoo')
#plt.legend(loc=4)
#plt.xlabel('time')
#plt.ylabel('Stock price (€)')

forecast_out = int(30) # predicting 30 days into future
df['Prediction'] = df[['Adj Close']].shift(-forecast_out) #  label column with data shifted 30 units up

# are there any NAN's in the data
print(df['Adj Close'].isnull().values.any())

# Calculate some features
dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

# Drop missing value
dfreg.fillna(value=-99999, inplace=True)

# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))

# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))

# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)

# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

# Separation of training and testing of model by cross validation train test split
# Here I'm not sure to use shuffle=False or not
# (see the dokumentation: https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation -
# 3.1.2.5. Cross validation of time series data)
# For timeseries data we must not shuffle the training / testing data for the correlation between time values -
# only when we asume uncorrelated values for each day we can treat all observations as independent and shuffle them.

# with shuffle - treating the stock price variables as uncorrelated between the days
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# without shuffle - treating the stock price variables as correlated between the days
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly4 = make_pipeline(PolynomialFeatures(3), Lasso())
clfpoly4.fit(X_train, y_train)

"""Test the model"""

confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test,y_test)
confidencepoly3 = clfpoly3.score(X_test,y_test)
confidencepoly4 = clfpoly4.score(X_test,y_test)

print(confidencereg,confidencepoly2,confidencepoly3,confidencepoly4)

# Printing the forecast; poly2 seemed most accurate.
forecast_set = clfpoly2.predict(X_lately)
#forecast_set = clfpoly3.predict(X_lately)
dfreg['Forecast'] = np.nan

"""Plotting the Prediction"""

last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]

dfreg['Adj Close'].tail(100).plot()
dfreg['Forecast'].tail(100).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
