# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:18:24 2020

This is a practice SVM strategy adapted from a quantitative finance blog. 

@author: Amogh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import talib as ta
import yfinance as yf
from sklearn import mixture as mix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = yf.download("IOC.NS", start = '2010-01-01', end = '2020-08-01')
df = df[['Open','High','Low','Close']]

# Deciding the lookback period for indicators and Training-testing split:
n = 10  #lookback period
t = 0.8
split = int(t*len(df))

# Preparing data and creating technical indicators:
df['high'] = df['High'].shift(1) 
df['low'] = df['Low'].shift(1) 
df['close'] = df['Close'].shift(1) 
df['RSI'] = ta.RSI(np.array(df['close']), timeperiod = n)
df['SMA'] = df['close'].rolling(n).mean()
df['Corr'] = df['SMA'].rolling(n).corr(df['close'])
df['SAR'] = ta.SAR(np.array(df['high']), np.array(df['low']), 0.2, 0.2)
df['ADX'] = ta.ADX(np.array(df['high']), np.array(df['low']), np.array(df['close']), timeperiod = n)
df['Corr'][df.Corr>1] = 1
df['Corr'][df.Corr<-1] = -1
df['Return'] = np.log(df['Open']/df['Open'].shift(1)) 
df = df.dropna()

# Creating Standard Scaler function and an Unsupervised Learning Algo for Regime prediction:
ss = StandardScaler()
unsup = mix.GaussianMixture(n_components=4, covariance_type="spherical", n_init=100, random_state=42)
df = df.drop(['High','Low','Close'], axis=1)
unsup.fit(np.reshape(ss.fit_transform(df[:split]),(-1,df.shape[1])))
regime = unsup.predict(np.reshape(ss.transform(df[split:]),(-1,df.shape[1])))

Regimes = pd.DataFrame(regime, columns=['Regime'], index=df[split:].index).join(df[split:], how='inner').assign(market_cu_return=df[split:].Return.cumsum()).reset_index(drop=False).rename(columns={'index':'Date'})

### PROBLEM WITH THE PLOT   XXXXXXXXXXXXXXXXXXXXXXXX                        
# Creating an order for Regimes and visualizing along with the means and covariances:
order = [0,1,2,3]
fig = sns.FacetGrid(data=Regimes, hue='Regime', hue_order=order, palette='Set1', aspect = 2, height = 4)
fig.map(plt.scatter,'Date','market_cu_return',s=4).add_legend()
plt.show()

for i in order:
    print('Mean for regime %i: '%i, unsup.means_[i][0])
    print('Co-variance for regime %i: '%i, (unsup.covariances_[i]))

# Scaling the Regimes dataframe for training the SVM and creating 'Signal' column for prediction:
ss1 = StandardScaler()
columns = Regimes.columns.drop(['Regime','Date'])
Regimes[columns] = ss1.fit_transform(Regimes[columns])
Regimes['Signal'] = 0
Regimes.loc[Regimes['Return']>0,'Signal'] = 1
Regimes.loc[Regimes['Return']<0,'Signal'] = -1
Regimes['return'] = Regimes['Return'].shift(1)
Regimes = Regimes.dropna()

# Instantiating the SVC classifer
clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, 
          degree=3, gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None,
          shrinking=True, tol=0.001, verbose=False)

# Splitting the data to fit the model:
split2 = int(0.8*len(Regimes))
X = Regimes.drop(['Signal','Return','market_cu_return','Date'], axis=1)
y = Regimes['Signal']
clf.fit(X[:split2],y[:split2])

p_data = len(X)-split2 #calculating test set size

# Saving the prediction column:
df['Pred_Signal'] = 0
df.iloc[-p_data:,df.columns.get_loc('Pred_Signal')] = clf.predict(X[split2:])
#print(df['Pred_Signal'][-p_data:])
df['str_ret'] = df['Pred_Signal']*df['Return'].shift(-1)

# Calculating cumulative returns of strategy and cumulative returns of market:
df['strategy_cu_return'] = 0
df['market_cu_return'] = 0
df.iloc[-p_data:,df.columns.get_loc('strategy_cu_return')] = np.nancumsum(df['str_ret'][-p_data:])
df.iloc[-p_data:,df.columns.get_loc('market_cu_return')] = np.nancumsum(df['Return'][-p_data:])
Sharpe = (df['strategy_cu_return'][-1]-df['market_cu_return'][-1])/np.nanstd(df['strategy_cu_return'][-p_data:])


# Visualization of stratgye vs market returns:
plt.plot(df['strategy_cu_return'][-p_data:], color='g', label='Strategy Returns')
plt.plot(df['market_cu_return'][-p_data:], color='r', label='Market Returns')
plt.figtext(0.14, 0.9, s='Sharpe Ratio: %.2f'%Sharpe)
plt.legend(loc='best')
plt.show()


