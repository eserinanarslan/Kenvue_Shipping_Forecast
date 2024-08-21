"""
File: model.py
Author: Eser Inan Arslan
Email: eserinanarslan@gmail.com
Description: Description: This file contains the code for running and forecasting with the model developed for Kenvue.
"""
#Import libraries
import numpy as np
import pandas as pd
import util
import preprocess
import xgboost as xgb
from math import sqrt
from pmdarima.arima import auto_arima

from statsmodels.tsa.seasonal import seasonal_decompose

from keras.models import Sequential
from keras.layers import Conv1D, LSTM, TimeDistributed, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

## ***XGBRegressor***
# Commented out IPython magic to ensure Python compatibility.
# # XGBRegressor Training
def XGB_modeller(train_X, train_y):
     parameters = { 'gamma' : [0, 0.5, 1], 'learning_rate' : [0.1, 0.15, 0.2, 0.25, 0.35, 0.4],
                   'max_depth' : [2, 5, 10, 15, 25],
                   'n_estimators' : [10, 25, 50, 75, 100],
                   'nthread' : [-1], 'reg_alpha' : [1], 'reg_lambda' : [1], 'seed' : [10] }

     bst = xgb.XGBRegressor()
     xgb_grid = GridSearchCV(bst,
                             parameters,
                             cv=5,
                             n_jobs=-1,
                             verbose=True,
                             )
     xgb_grid.fit(train_X, train_y, eval_set=[(train_X, train_y)], early_stopping_rounds=50)
     return xgb_grid

## ***Auto Arima***
# # Arima Training
def auto_arima_modeller(arima_train):
    arima_stepwise_model = auto_arima(arima_train, start_p=0, start_q=0, max_p=13, max_q=13, m=12,
                                      start_P=0, seasonal=True, d=1, D=1, trace=True,
                                      error_action='ignore', suppress_warnings=True, stepwise=True,
                                      n_jobs=-1) #n_jobs for parallel process


    # Fit arima model
    arima_predicts = arima_stepwise_model.fit(arima_train)

    return arima_predicts

# **Long Short - Term Memory (LSTM)**

def lstm_modeller(train_X_lstm, train_Y_lstm, test_X_lstm, test_Y_lstm):
    # define parameters
    verbose, epochs, batch_size = 1, 100, 10
    n_timesteps, n_features, n_outputs = train_X_lstm.shape[1], train_X_lstm.shape[2], train_Y_lstm.shape[1]

    n_timesteps, n_features, n_outputs

    # define model
    lstm_model = Sequential()
    lstm_model.add(Conv1D(filters=32, kernel_size=5,
                          strides=1, padding="causal",
                          activation="relu",
                          input_shape=(n_timesteps, n_features)))

    lstm_model.add(LSTM(200, activation='relu', return_sequences=True))
    # lstm_model.add(RepeatVector(n_outputs))
    lstm_model.add(LSTM(200, activation='relu', return_sequences=True))
    lstm_model.add(Dropout(0.2))  # Adding Dropout layer
    lstm_model.add(TimeDistributed(Dense(1, activation='relu')))
    #lstm_model.add(TimeDistributed(Dense(1)))
    #lstm_model.compile(loss='mse', optimizer=Adam(lr=0.001))
    lstm_model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['acc'])

    history = lstm_model.fit(train_X_lstm, train_Y_lstm, epochs=epochs, batch_size=batch_size, validation_data=(test_X_lstm, test_Y_lstm),
                             callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=verbose, shuffle=False)

    lstm_model.summary()

    return  lstm_model, verbose


def main():
    # Read configuration from the config file
    config_path = "config.ini"
    config = util.read_config(config_path)

    # Prepare training data for model
    print(" Data is ready for preprocess")
    (supervised_train, supervised_test, train_X_lstm, test_X_lstm, train_Y_lstm, test_Y_lstm, scaler ,
     delivery_count, train_x, train_y, train_scaled, test_scaled) = preprocess.main()

    # Call training models
    print(" Data is ready for model")
    result_df = train_model(train_x, train_y, delivery_count, train_X_lstm, train_Y_lstm, test_X_lstm, test_Y_lstm, test_scaled, scaler)

    # Get values from the config file
    output_path = config.get("Settings", "results_path")

    # Export dataframe as a csv
    result_df.to_csv(output_path, index=0)

def train_model(train_x, train_y, delivery_count, train_X_lstm, train_Y_lstm, test_X_lstm, test_Y_lstm, test_scaled, scaler):
    xgb_grid = XGB_modeller(train_x, train_y)

    # Invert scale predictions to time series
    predictions = list()
    start = 7  # test period
    l = len(test_scaled) - start

    rmse = []
    mape = []

    for i in range(len(test_scaled)):
        X1, y = test_scaled[l, 0:-1], test_scaled[l, -1]
        X1 = X1[-7:]  # ts_window

        # prediction
        # pred = predict()
        X1 = X1.reshape((1, -1))
        pred = xgb_grid.predict(X1)

        yhat = util.invert_scale(scaler, X1[0], pred)

        l = l + 1

        predictions.append(yhat)

        rmse.append(sqrt(mean_squared_error([delivery_count['Count'][-7:][i]], [yhat])))
        mape.append(util.mean_absolute_percentage_error([delivery_count['Count'][-7:][i]], [yhat]))

    print("Test RMSE:", np.mean(rmse))
    print("Test MAPE:", np.mean(mape))

    result_df = delivery_count[-7:]
    result_df['XGB_Predictions'] = predictions

    result = seasonal_decompose(delivery_count, model='multiplicative')
    fig = result.plot()

    arima_train = delivery_count[:-7]

    arima_test = delivery_count[-7:]

    arima_train['Count'] = util.fill_nan_with_neighbors(arima_train['Count'])

    arima_train['Count'] = util.clip_outliers(arima_train['Count'], sigma=1.5)
    ###
    arima_train = arima_train.fillna(arima_train.mean())  # Fill NaN values with the mean
    ###
    arima_predicts = auto_arima_modeller(arima_train)

    # Predict results with auto-arima

    future_forecast = arima_predicts.predict(n_periods=12)

    arima_rmse = sqrt(mean_squared_error(delivery_count['Count'][-7:], future_forecast))

    arima_mape = util.mean_absolute_percentage_error(delivery_count['Count'][-7:], future_forecast)

    print("Test RMSE:", np.mean(arima_rmse))
    print("Test MAPE:", np.mean(arima_mape))

    result_df['Arima_Predictions'] = future_forecast

    lstm_model, verbose = lstm_modeller(train_X_lstm, train_Y_lstm, test_X_lstm, test_Y_lstm, scaler)

    # Predict results with LSTM Model
    lstm_predicts = lstm_model.predict(test_X_lstm, verbose=verbose)

    # Invert scaled predictons to time series
    predictions = list()
    start = 7  # test period
    l = len(test_scaled) - start

    lstm_rmse = []
    lstm_mape = []

    for i in range(len(test_scaled)):
        X1, y = test_scaled[l, 0:-1], test_scaled[l, -1]
        X1 = X1[-7:]  # ts_window

        X1 = X1.reshape((1, -1))

        yhat = util.invert_scale(scaler, X1[0], lstm_predicts[i][0])

        l = l + 1

        predictions.append(yhat)

        lstm_rmse.append(sqrt(mean_squared_error([delivery_count['Count'][-7:][i]], [yhat])))
        lstm_mape.append(util.mean_absolute_percentage_error([delivery_count['Count'][-7:][i]], [yhat]))

    print("LSTM RMSE:", np.mean(lstm_rmse))
    print("LSTM MAPE:", np.mean(lstm_mape))

    result_df['LSTM_Predictions'] = predictions

    result_df.plot(figsize=(10, 4))

    # Export dataframe as a csv
    result_df.to_csv("results.csv", index=0)

    print('Actual count :', result_df.Count.sum())
    print('XGB count :', result_df.XGB_Predictions.sum())
    print('ARIMA count :', result_df.Arima_Predictions.sum())
    print('LSTM count :', result_df.LSTM_Predictions.sum())

    return result_df

if __name__ == '__main__':
    # Set parameters to see all data
    pd.set_option('display.max_rows', 150)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 1000)

    main()
