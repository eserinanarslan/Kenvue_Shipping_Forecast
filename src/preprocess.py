"""
File: preprocess.py
Author: Eser Inan Arslan
Email: eserinanarslan@gmail.com
Description: Description: This file contains the code for running and forecasting with the model developed for Kenvue.
"""
import util

import pandas as pd
import numpy as np


# Main method
def main():
    # Read configuration from the config file
    config_path = "config.ini"
    config = util.read_config(config_path)
    # Get values from the config file
    data_path = config.get("Settings", "data_path")
    df = util.read_data(data_path)

    df = util.remove_whitespace(df)
    print('White spaces cleaned!!!')

    # Convert date columns
    date_columns = config.get("params", "date_column_list").split(', ')
    for col in date_columns:
        print(col)
        df[col] = pd.to_datetime(df[col])

    # Detect null values in 'Date' columns
    for col in date_columns:
        print(col)
        null_dates = df[df[col].isnull()]
        print(null_dates.shape)
        print('*************')
    # null_value_list = config.get("params", "null_value_list")
    # white_space_list = config.get("params", "white_space_list")
    util.missing_value_analyse(df)
    # Missing value imputation
    df['ShippingConditionCode'] = df['ShippingConditionCode'].replace(' ', '0')
    df['ShippingPlantNumber'] = df['ShippingPlantNumber'].replace(' ', 'AA00')
    df['InitialCommercialCode'] = df['InitialCommercialCode'].replace(' ', 'Unknown')
    df['SalesOrganizationNumber'] = df['SalesOrganizationNumber'].replace(' ', 'AA00')
    df['GroupCode'] = df['GroupCode'].replace(' ', '0000')

    # Calculating delivery counts per day
    delivery_count = df['DeliveryDateTime'].value_counts().reset_index()
    delivery_count = delivery_count.sort_values(by='DeliveryDateTime', ascending=True)
    delivery_count.columns = ['DeliveryDateTime', 'Count']

    delivery_count.set_index('DeliveryDateTime', inplace=True)

    delivery_count = delivery_count.sort_index()

    # Plot data
    delivery_count.iloc[:200].plot(figsize=(20, 8))

    # Stationary Control
    s_test = util.Stationary()

    s_test.ADF_Stationarity_Test(delivery_count['Count'])
    print(s_test.isStationary)

    diff_values = pd.Series(delivery_count['Count'])
    diff_values = util.clip_outliers(diff_values, sigma=1.5)

    # split train test datasets
    xtrain, xtest = diff_values[0:-7], diff_values[-7:]

    # reorganize dataset according to window size
    values_unscaled = np.concatenate((xtrain, xtest))
    supervised_raw = util.timeseries_to_supervised(values_unscaled, 7)

    supervised_raw = supervised_raw.values.astype("float32")

    scaler, train_scaled, test_scaled = util.scale_value(xtrain.values.reshape(len(xtrain), 1),
                                                         xtest.values.reshape(len(xtest), 1))

    values_scaled = np.concatenate((train_scaled, test_scaled))

    supervised = util.timeseries_to_supervised(values_scaled, 7)

    supervised_values = supervised.values.astype('float32')

    # split supervised data into train and test-sets
    supervised_train, supervised_test = supervised_raw[0:-7], supervised_raw[-7:]
    train_scaled, test_scaled = supervised_values[0:-7], supervised_values[-7:]

    train_x, train_y = train_scaled[:, :-1], train_scaled[:, -1]
    test_x, test_y = test_scaled[:, :-1], test_scaled[:, -1]

    train_y.reshape(train_y.shape[0], 1)

    # prepare train dataset for lstm
    train_x_lstm = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x_lstm = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

    # prepare test dataset for lstm
    train_y_lstm = train_y.reshape((train_y.shape[0], 1, 1))
    test_y_lstm = test_y.reshape((test_y.shape[0], 1, 1))

    return (supervised_train, supervised_test,
            train_x_lstm, test_x_lstm, train_y_lstm, test_y_lstm, scaler,
            delivery_count, train_x, train_y, train_scaled, test_scaled)


if __name__ == '__main__':
    main()
