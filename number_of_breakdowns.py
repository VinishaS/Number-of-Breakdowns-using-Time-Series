#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.tsa.api as smt


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[5]:


df = pd.read_excel(r"C:\Users\Vinisha Sunkara\Downloads\break_down_data_2015to2018.xls")
df


# In[6]:



data = df[['BD_DATE','DEPT_CODE']]
data['BD_DATE']=pd.to_datetime(data['BD_DATE'],errors='coerce')
data = data.sort_values('BD_DATE')
data = data.dropna()
data.head()


# In[7]:


df[['BD_DATE','DEPT_CODE']]=data
df.head()


# In[9]:


condition = pd.to_datetime('2014-12-31')
data = data[(data['BD_DATE']>condition)]
data.reset_index(drop=True,inplace=True)


# In[10]:


data.set_index('BD_DATE',inplace=True)
final_data = data.resample('D').count()
final_data.columns =['BD_COUNT']


# In[11]:


df.info()


# In[12]:


plt.figure(figsize=(10,6))
plt.ylim(0,50)
plt.scatter(final_data.index,final_data['BD_COUNT'])


# In[13]:



from scipy.stats.mstats import winsorize
winsorized_data = winsorize(final_data['BD_COUNT'], limits=[0.01, 0.01])
processed_data = final_data.copy()
processed_data['BD_COUNT'] = winsorized_data


# In[14]:


from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess


# In[15]:


def test_stationarity (timeseries) :
    #Determing rolling statistics
    MA = timeseries.rolling(window=365).mean()
    MSTD = timeseries.rolling(window=365).std()
    #plot rolling statistics: 
    plt.figure(figsize=(15,5))
    orig = plt.plot(timeseries, color='pink', label = 'original')
    mean = plt.plot(MA, color='blue', label = 'mean')
    std = plt.plot(MSTD, color='black', label = 'Rolling std')
    plt.legend (loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '"Lags Used', 'Number of observations Used' ])
    for key, value in dftest[4].items():
        dfoutput['Critical value (%s) '%key] = value
    print(dfoutput)


# In[16]:


def tsplot (y, lags=None, figsize=(15, 5), style='bmh'):
    if not isinstance(y, pd.Series) :
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        act_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p=(0:.5f)'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=act_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()


# In[17]:


test_stationarity(processed_data)

processed_data


# In[18]:


trained_data = pd.DataFrame(processed_data[:1200])
test_data = pd.DataFrame(processed_data[1200:])
test_data.head()


# In[19]:


test_stationarity(trained_data)


# In[20]:


dec = sm.tsa.seasonal_decompose(trained_data['BD_COUNT'], model = 'additive').plot()
plt.show()


# In[21]:


tsplot(trained_data['BD_COUNT'])


# In[22]:


from statsmodels.tsa.arima.model import ARIMA


# In[23]:


model = ARIMA(trained_data['BD_COUNT'], order=(5,0,5))
model_fit = model.fit()
print(model_fit.summary())


# In[24]:


train_pred = model_fit.predict(start=trained_data.index[0], end=trained_data.index[-1])
mae = mean_absolute_error(trained_data['BD_COUNT'], train_pred)
mse = mean_squared_error(trained_data['BD_COUNT'], train_pred)
print('MAE :' , mae )
print('MSE :', mse)

trained_data.plot()
train_pred.plot()
plt.legend()


# In[25]:


test_pred = model_fit.predict(start=test_data.index[0], end=test_data.index[-1])
mae = mean_absolute_error(test_data['BD_COUNT'], test_pred)
mse = mean_squared_error(test_data['BD_COUNT'], test_pred)
print('MAE :' , mae )
print('MSE :', mse)

test_data.plot()
test_pred.plot()
plt.legend()


# In[26]:


plot_predict(model_fit,1,1400)
plt.show()


# In[27]:


import pickle
with open('arima_model.pkl', 'wb') as file:
    pickle.dump(model_fit, file)


# In[28]:


get_ipython().system('pip install streamlit')


# In[ ]:




