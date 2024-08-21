"""
File: util.py
Author: Eser Inan Arslan
Email: eserinanarslan@gmail.com
Description: Description: This file contains the code for running and forecasting with the model developed for Kenvue.
"""

import pandas as pd
import numpy as np
import configparser

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller



# Drop whitespace of column names
def remove_whitespace(df):
    df.columns = df.columns.str.strip().str.replace(' ', '')

    # return new dataframe and new column names
    return df

# Missing Value Analyse
def missing_value_analyse(df):
  col_list = df.columns.to_list()
  for col in col_list:
    print('Null values for ', col)
    null_values = df[df[col].isnull()]
    print('Shape of ', col, 'is : ', null_values.shape)

    print('Whitespace values for ', col)
    print('Shape of ', col, 'is : ', df[df[col] == ' '].shape)

    print('*************')

#Stationary Control
#if P-Value is smaller than 0.005, we can say that there is stationary, else there is no stationary
#if our data has no stationary, I will plan to use differences
class Stationary:
    def __init__(self, significance=.005):
        self.SignificanceLevel = significance
        self.pValue = None
        self.isStationary = None

    def ADF_Stationarity_Test(self, timeseries, printResults=True):

        # Dickey-Fuller test:
        print(timeseries)
        try:
            adfTest = adfuller(timeseries, autolag='AIC')

            self.pValue = adfTest[1]

            if (self.pValue < self.SignificanceLevel):
                self.isStationary = True
            else:
                self.isStationary = False
            #self.isStationary=True
            if printResults:
                dfResults = pd.Series(adfTest[0:4],
                                      index=['ADF Test Statistic', 'P-Value', '# Lags Used', '# Observations Used'])

                # Add Critical Values
                for key, value in adfTest[4].items():
                    dfResults['Critical Value (%s)' % key] = value

                print('Augmented Dickey-Fuller Test Results:')
                print(dfResults)
        except Exception as e:
          print(e)
          self.isStationary = False

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        # print(interval, dataset[i], dataset[i - interval])
        diff.append(value)
    return pd.Series(diff)


# Limits the outliers in the given series based on a specified sigma value, 1.5
# and returns Pandas Series, the clipped data.
def clip_outliers(series, sigma=1.5):
    series_mean = series.mean()
    series_std = series.std()
    lower_bound = series_mean - sigma * series_std
    upper_bound = series_mean + sigma * series_std
    return series.clip(lower=lower_bound, upper=upper_bound)

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    dfx = pd.DataFrame(data)
    df = dfx.assign(**{
        '{} (t-{})'.format(col, t): dfx[col].shift(t)
        for t in range(lag+1)
        for col in dfx
    })

    df=df.drop([df.columns[0]], axis=1)
    df=df[df.columns[::-1]]
    return df[lag:]

# scale train and test data to [-1, 1] with MinMaxScaler
def scale_value(train, test):
    # create scaler
    scaler = MinMaxScaler()

    # fit scaler
    scaler = scaler.fit(train)

    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    tr_scaled = scaler.transform(train)

    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    ts_scaled = scaler.transform(test)

    return scaler, tr_scaled, ts_scaled

# Read dataset
def read_data(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=',')
    except pd.errors.ParserError as e:
        print(f'Error while parsing CSV file: {e}')
    return df


# Inverts the scaling transformation applied to a value.
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    # Ensure all elements in new_row are of the same data type
    new_row = [float(x) for x in new_row]  # Convert to float if needed
    array = np.array(new_row)
    array = array.reshape(1, len(array))

    return scaler.inverse_transform(array)[0, -1]


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# Error measurement metrics
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / y_true)) * 100

# Read config file
def read_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

# Fills NaN values with the average of the previous and next records.
def fill_nan_with_neighbors(series):

    filled_series = series.copy()

    for i in range(len(series)):
        # If the value is NaN
        if pd.isnull(series[i]):
            # Get the previous and next values
            prev_value = series[i - 1] if i > 0 else None
            next_value = series[i + 1] if i < len(series) - 1 else None

            # Calculate the average
            if prev_value is not None and next_value is not None:
                filled_series[i] = (prev_value + next_value) / 2
            elif prev_value is not None:
                filled_series[i] = prev_value
            elif next_value is not None:
                filled_series[i] = next_value
            else: # If both previous and next values are missing, fill with 0
                filled_series[i] = 0

    return filled_series

# Checks for NaN values in a pandas Series.
def nan_check(series):
    has_nan = series.isnull().any()
    return has_nan

# Finds the indices of NaN values in a pandas Series.
def find_nan_indices(series):
    nan_indices = series.index[series.isnull()].tolist()
    return nan_indices


