# Shipping - Forecast

The task is to forecast the shipping of next week(7 days)

# Project Summary

## Introduction
This project aimed to develop a time-dependent solution for forecasting the total number of deliveries made per day. Despite the possibility of solving the problem using regression methods, I chose to focus on developing a time series solution.

## Initial Forecasting Attempts
I initially attempted to forecast the total deliveries per day using three different algorithms: XGBRegressor, ARIMA, and LSTM. However, the results obtained from these algorithms were not satisfactory.

## Feature Analysis
Subsequently, I began analyzing other parameters to use them as additional variables. Using correlation analysis, I calculated the correlation between features and the target value. The results indicated the following cumulative scores for each category:

- GroupCode: 0.326193
- ShippingPlantNumber: 0.293258
- SalesOrganizationNumber: 0.318402
- DeliveryTypeCode: 0.218895
- InitialCommercialCode: 0.249634
- ShippingConditionCode: 0.039766

Based on these scores, the categories with the highest total scores were GroupCode, SalesOrganizationNumber, and ShippingPlantNumber.

## Challenges Faced
However, the categorical nature of the data and its diversity led to the dataset becoming sparse, posing challenges in achieving a satisfactory accuracy rate within reasonable computational time.

## Execution
In this solution case, you can execute training independently. For forecast success measurement criteria, Mean absolute percentage error and root mean squared error were used. However, results are under satisfaction level.

In this solution, 3 different machine learning algorithms were used.

Again after forecast, you can create rest api to see results. "main.py" folder was created for rest service. In this step for easy and fast execution, I prefer to dockerize the service. For dockerization, you have to run below commands on terminal.

*** For model training, you have to run "python src/model.py" on terminal

*** For model service, you have to run "python main.py" on terminal

But I highly recommend to use dockerize flask service version with help of below shell scripts

1) docker build --tag shipping-forecast-app:1.0 .
2) docker run -p 1001:1001 --name shipping-forecast-app shipping-forecast-app:1.0

## Service

After training and forecasting process, you can use Postman to test. You can find postman file under "collection" file. You have to import that json file to the Postman. 


(get_all_results) : This service return possible values for every day. This method doesn't need any parameter. 

Services return dataframe as a json message.
