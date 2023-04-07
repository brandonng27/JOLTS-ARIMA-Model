import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller

#Retrieve JOLTS data, rename column, change date to date-time format, resample data
jolts = pd.read_csv("JTSJOL-2.csv")
jolts.rename(columns={'JTSJOL': 'JOLTS'}, inplace=True)
jolts['DATE'] = pd.to_datetime(jolts['DATE'])
indexed_jolts = jolts.set_index(['DATE'])
indexed_jolts = indexed_jolts.resample('MS').mean()

print("Real GDP data is chosen to be included in the model as I hypothesize that there is a high correlation between JOLTS and Real GDP data as during downturns of the economy, there would be less job openings as companies hire less.\n")

#Retrieve quarterly Real GDP data from FRED, rename column, change date to date-time format
rgdp = pd.read_csv("GDPC1.csv")
rgdp.rename(columns={'GDPC1': 'RGDP'}, inplace=True)
rgdp['DATE'] = pd.to_datetime(rgdp['DATE'])

#Resample to monthly frequency with date being at month start to match the date of JOLTS data, use forward fill to fill in missing values
indexed_rgdp = rgdp.set_index(['DATE'])
monthly_rgdp = indexed_rgdp.resample('MS').ffill()

#Merge data, use both forward and backward fill to fill in missing data
data = pd.concat([indexed_jolts, monthly_rgdp], axis=1)
data = data.fillna(method="ffill")
data = data.fillna(method="bfill")

#Check for the stationarity of the data using rolling statistics, and Augmented Dicky-Fuller Test.
def test_stationarity(timeseries, column):
    
    #Determine rolling statistics
    movingAverage = timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()
    
    #Plot rolling statistics
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(movingAverage, color='red', label='Rolling Mean')
    std = plt.plot(movingSTD, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey–Fuller test:
    print('\nResults of Dickey Fuller Test:')
    dftest = adfuller(timeseries[column], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(data, 'JOLTS')

print("\nWe can observe that there is a trend component to the rolling mean for both jolts and rgdp data, while the rolling standard deviation is relatively constant with time.")
print("\nFrom the Augumented Dicky-Fuller Test, the p-value is very high and the critical values at 1%, 5%, and 10% are not close to the test statistic, so we can conclude that the data is not stationary.")

print("\nTo achieve stationarity, we estimate the trend using log scale and use the difference between the log scale and its moving average\n")

#Estimate trend using log scale
data_logScale = np.log(data)

#We use the difference between the log scale data and its moving average to make the data stationary
movingAverage = data_logScale.rolling(window=3).mean()
datasetLogScaleMinusMovingAverage = data_logScale - movingAverage
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
test_stationarity(datasetLogScaleMinusMovingAverage, 'JOLTS')

print("\nFrom the plot, we can observe that the trend component is now removed, and from the Augmented Dicky-Fuller Test, the p-value is now much lower, and the critical values at 1%, 5%, and 10% are now close to the test statistic, so the data is now stationary.\n")
print("We can now plot the ACF and the PACF to determine the order of the ARIMA model\n")
#ACF & PACF plots
lag_acf = acf(datasetLogScaleMinusMovingAverage['JOLTS'], nlags=50)
lag_pacf = pacf(datasetLogScaleMinusMovingAverage['JOLTS'], nlags=50, method='ols')

#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogScaleMinusMovingAverage)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogScaleMinusMovingAverage)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')            

#Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogScaleMinusMovingAverage)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogScaleMinusMovingAverage)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
            
plt.tight_layout()
plt.show()

print("\nAs both the ACF and PACF shows a sharp drop, a ARIMA(p,d,q) model should be appropriate\n")
print("The lag before the drop in the PACF indicates the value of p, in this case it is 1, we assume that p = q first and refine it later\n")
print("The value of d would be determined by experimenting with different values and choosing the one with the best fit\n")

#Split data into train and test sets, 80% training set, 20% testing set
trainsize = int(0.8*len(datasetLogScaleMinusMovingAverage))
testsize = len(datasetLogScaleMinusMovingAverage) - trainsize
train_data = datasetLogScaleMinusMovingAverage.iloc[:trainsize]
test_data = datasetLogScaleMinusMovingAverage.iloc[trainsize:]

print("Plotting the ARIMA model")
#We build the ARIMA model
model = ARIMA(train_data['JOLTS'], order=(1,0,1))
results_ARIMA = model.fit()
plt.plot(datasetLogScaleMinusMovingAverage['JOLTS'], color='blue', label='Original')
plt.plot(results_ARIMA.predict(start='2001-09-01', end='2018-10-01'), color='red', label='ARIMA fit training data')
plt.legend(loc='best')
plt.title('ARIMA model')
plt.show()

print("\nWe make predictions for the test data, and reconvert our data back to its original form\n")

#Adding back its moving average
arima_LogScaleMinusMovingAverage = results_ARIMA.predict(start='2018-11-01', end='2023-02-01')
arima_LogScaleMinusMovingAverage = pd.DataFrame(arima_LogScaleMinusMovingAverage)
arima_LogScaleMinusMovingAverage.columns = ['JOLTS']
movingAverage_filled = movingAverage.loc['2018-11-01':'2023-02-01'].fillna(method='ffill')
arima_LogScale = arima_LogScaleMinusMovingAverage['JOLTS'] + movingAverage_filled['JOLTS']
#Inversing the log scale
arima_Original = np.exp(arima_LogScale)

print("Plotting the ARIMA predictions against the test data")
plt.plot(data['JOLTS'], color='blue', label='Original')
plt.plot(arima_Original, color='red', label='ARIMA prediction test data')
plt.legend(loc='best')
plt.title('ARIMA prediction')
plt.show()

print("\nEvaluating the ARIMA model using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)\n")
#Extract the actual values and ARIMA predictions for the test set
jolts_true = data['JOLTS'].loc['2018-11-01':'2023-02-01']
jolts_true = jolts_true.values
jolts_pred = arima_Original.values

#Calculate the root mean squared error (RMSE)
rmse = mean_squared_error(jolts_true, jolts_pred, squared=False)
print("ARIMA RMSE: ", rmse)

#Calculate the mean absolute error (MAE)
mae = mean_absolute_error(jolts_true, jolts_pred)
print("ARIMA MAE: ", mae)

print("\nBothe the RMSE and MAE values are very low, indicating that the ARIMA model that I built is a good model for JOLTS data")
print("\nSummary of model and findings:\n")
print('''This study aimed to build a predictive model for JOLTS data.
To begin, the JOLTS data was retrieved and prepared for analysis.
Real GDP was chosen as the input variable due to its positive correlation with JOLTS data.\n
Initially, rolling statistics and the Augmented Dickey-Fuller Test were employed to determine the stationarity of the data.
Since the data was found to be non-stationary, the log scale minus the moving average of the data was used.\n
After confirming the stationarity of the data, the ACF and PACF plots were generated to identify the optimal order of the ARIMA model.
The data was then split into training and testing sets, and the model was developed using the training data.\n
The trained model was used to predict the test data, which was converted back into its original form by adding back the moving average and taking the exponential of the data.
The accuracy of the model was evaluated using the RMSE and MAE, which yielded a low error rate and indicated a good model fit for predicting the test data.''')



