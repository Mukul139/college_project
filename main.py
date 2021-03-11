# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 21:59:06 2021

@author: mukul
"""

import pandas as pd
import numpy as np
dataset = pd.read_csv("pollution data/pollution_data_delhi.csv")
dataset = dataset.iloc[731:,:]
dataset = dataset.drop('Chandni Chowk, Delhi - IITM', 1)

def get_index(data,value):
    return data.loc[data['Parameter']==value].index[0]

def pollutant_average(pollutant,pollutant_name):
    try:
        avg = pollutant.loc[get_index(pollutant,pollutant_name),'Average']
        if avg!='NA':
            avg = float(avg)
        else:
            avg = np.nan
        
        return avg
    except IndexError:
        return np.nan

station_pollutant_mean = pd.DataFrame(index=('O3','NOx','CO','SO2','PM10','PM2.5','Minor'))

station_pollutant_dict={}

def cal_station_pollution(station_data,station_name):
    station_pollutant = pd.DataFrame(columns=('Date','O3','NOx','CO','SO2','PM10','PM2.5','Minor'))
    for i in range(len(station_data)):
       
        station_pollutant.loc[i,'Date'] = station_data[i,0]
        if str(station_data[i,1])!='nan':
            try:
                pollutant = pd.read_csv(f'pollution data/pollution_data/{station_name}/{station_data[i,1]}.csv')
                try:
                    pollutant['Average'] = pollutant['Average'].str.strip()
                    pollutant['Parameter'] = pollutant['Parameter'].str.strip()
                except AttributeError:
                    pass
                except KeyError as e:
                    print(e)
                    continue
                station_pollutant.loc[i,'O3'] = pollutant_average(pollutant,'Ozone')
                station_pollutant.loc[i,'NOx'] = pollutant_average(pollutant,'NOx')
                station_pollutant.loc[i,'CO'] = pollutant_average(pollutant,'CO')
                station_pollutant.loc[i,'SO2'] = pollutant_average(pollutant,'SO2')
                station_pollutant.loc[i,'PM10'] = pollutant_average(pollutant,'PM10')
                station_pollutant.loc[i,'PM2.5'] = pollutant_average(pollutant,'PM2.5')
                
                minor_pollutant_name = ['Benzene','Eth Benzene','MP Xylene','NH3','NO','NO2','Toluene','Xylene']
                minor_pollutant = 0
                for name in minor_pollutant_name:
                    try:
                        index = get_index(pollutant,name)
                    except IndexError:
                         continue
                    if str(pollutant.loc[index,'Average'])!='NA':
                        pol=float(pollutant.loc[index,'Average'])
                        if pol<0:
                            continue
                        minor_pollutant+=pol
                        
                    
                
                station_pollutant.loc[i,'Minor'] = minor_pollutant
                
                
                
            except FileNotFoundError:
                pass
    return station_pollutant


for station_name in dataset.columns:
    if station_name=='date':
        continue
    station_data=dataset.loc[:,('date',station_name)].values
    station_data = np.flipud(station_data)

    
    station_pollutant = cal_station_pollution(station_data,station_name)
    station_pollutant = station_pollutant.astype({'O3':float,'NOx':float,'CO':float,'SO2':float,'PM10':float,'PM2.5':float,'Minor':float})
    station_pollutant.Date = pd.to_datetime(station_pollutant.Date)
    station_pollutant_mean.loc['O3',station_name]=station_pollutant[['O3']].apply(np.mean).mean()
    station_pollutant_mean.loc['NOx',station_name]=station_pollutant[['NOx']].apply(np.mean).mean()
    station_pollutant_mean.loc['CO',station_name]=station_pollutant[['CO']].apply(np.mean).mean()
    station_pollutant_mean.loc['SO2',station_name]=station_pollutant[['SO2']].apply(np.mean).mean()
    station_pollutant_mean.loc['PM10',station_name]=station_pollutant[['PM10']].apply(np.mean).mean()
    station_pollutant_mean.loc['PM2.5',station_name]=station_pollutant[['PM2.5']].apply(np.mean).mean()
    station_pollutant_mean.loc['Minor',station_name]=station_pollutant[['Minor']].apply(np.mean).mean()
    # print(station_pollutant.Minor.mean())
    
    station_pollutant_dict[station_name] = station_pollutant

# station_pollutant_dict['Aya Nagar, Delhi - IMD'].Minor.mean()

import matplotlib.pyplot as plt
import seaborn as sns 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf


sns.heatmap(station_pollutant_mean, cmap ='RdYlGn', linewidths = 0.30, annot = True) 
# =============================================================================
# Calculate value average value for Delhi
# Heat map
# =============================================================================
delhi_plotting_data = pd.DataFrame(columns=('Date','O3','NOx','CO','SO2','PM10','PM2.5','Minor'))
# delhi_plotting_data = station_pollutant_dict[]
df0= pd.DataFrame()
for key in station_pollutant_dict:

    # delhi_plotting_data = pd.concat([delhi_plotting_data, station_pollutant_dict[key]], axis=0).groupby('O3').sum().reset_index()

    df0 = pd.concat([df0, station_pollutant_dict[key]], axis=1, ) 

delhi_plotting_data['O3']=df0.loc[:, 'O3'].mean(axis=1)
delhi_plotting_data['NOx']=df0.loc[:, 'NOx'].mean(axis=1)
delhi_plotting_data['CO']=df0.loc[:, 'CO'].mean(axis=1)
delhi_plotting_data['SO2']=df0.loc[:, 'SO2'].mean(axis=1)
delhi_plotting_data['PM10']=df0.loc[:, 'PM10'].mean(axis=1)
delhi_plotting_data['PM2.5']=df0.loc[:, 'PM2.5'].mean(axis=1)
delhi_plotting_data['Minor']=df0.loc[:, 'Minor'].mean(axis=1)
delhi_plotting_data['Date']=df0.loc[:, 'Date']

plotting_data=delhi_plotting_data.dropna(axis=0)

plotting_data.Date = pd.to_datetime(plotting_data.Date)

plotting_data = plotting_data.astype({'O3':float,'NOx':float,'CO':float,'SO2':float,'PM10':float,'PM2.5':float,'Minor':float})

plotting_data.info()

plotting_data.describe()

plotting_data = plotting_data.sort_values(by='Date') 

# Histogram of all numeric fields
df_hist = plotting_data.drop(columns=['Date'],axis=1)
df_hist.hist(figsize=(15,15))

# Visualise the target variable
plt.plot(plotting_data['Date'], plotting_data['O3'], label='O3')

plt.plot(plotting_data['Date'], plotting_data['NOx'], label='NOx')

plt.plot(plotting_data['Date'], plotting_data['CO'], label='CO')

plt.plot(plotting_data['Date'], plotting_data['SO2'], label='SO2')

plt.plot(plotting_data['Date'], plotting_data['PM10'], label='PM10')

plt.plot(plotting_data['Date'], plotting_data['PM2.5'], label='PM2.5')

plt.plot(plotting_data['Date'], plotting_data['Minor'], label='Minor pollutant')
plt.xlabel('Date')
plt.ylabel('pollution ug/m3')
plt.title('Delhi pollution',)
plt.legend()
plt.show()


def plot(plotting_data,pollutant_name):
    df_series = plotting_data.set_index('Date')
    series = pd.Series(df_series[pollutant_name], index= df_series.index)
    results = seasonal_decompose(series, model='additive',freq = len(series)//2)
    plt.rcParams['figure.figsize'] = (20.0, 10.0)
    results.plot()
    plt.show()
    
    #calling acf function from stattools
    lag_acf = acf(series, nlags=50)
    plt.figure(figsize=(16, 7))
    #Plot ACF: 
    plt.plot(lag_acf, marker="o")
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(series)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(series)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    plt.xlabel('number of lags')
    plt.ylabel('correlation')
    plt.tight_layout()
    
    
    #calling pacf function from stattool
    lag_pacf = pacf(series, nlags=50, method='ols')
    
    #PLOT PACF
    plt.figure(figsize=(16, 7))
    plt.plot(lag_pacf, marker="o")
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(series)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(series)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.xlabel('number of lags')
    plt.ylabel('correlation')
    plt.tight_layout()

plot(plotting_data,'O3')
plot(plotting_data,"NOx")
plot(plotting_data,'CO')
plot(plotting_data,'SO2')
plot(plotting_data,'PM10')
plot(plotting_data,'PM2.5')
plot(plotting_data,'Minor')




#Use linear interpolation to fill up nulls
station_pollutant.Date = pd.to_datetime(station_pollutant.Date)

station_pollutant = station_pollutant.astype({'O3':float,'NOx':float,'CO':float,'SO2':float,'PM10':float,'PM2.5':float,'Minor':float})

station_pollutant.info()

df = station_pollutant.interpolate(method='linear', axis=0).ffill().bfill()
df.head(10)
# =============================================================================
# 
# 
# # Outlier Detection using Inter Quartile Range
# def out_iqr(s, k=1.5, return_thresholds=False):
#     """
#     Return a boolean mask of outliers for a series
#     using interquartile range, works column-wise.
#     param k:
#         some cutoff to multiply by the iqr
#     :type k: ``float``
#     param return_thresholds:
#         True returns the lower and upper bounds, good for plotting.
#         False returns the masked array 
#     :type return_thresholds: ``bool``
#     """
#     # calculate interquartile range
#     q25, q75 = np.percentile(s, 25), np.percentile(s, 75)
#     iqr = q75 - q25
#     # calculate the outlier cutoff
#     cut_off = iqr * k
#     lower, upper = q25 - cut_off, q75 + cut_off
#     if return_thresholds:
#         return lower, upper
#     else: # identify outliers
#         return [True if x < lower or x > upper else False for x in s]
#     
#     
# # For comparison, make one array each at varying values of k.
# df1 = df.drop(columns=['Date'],axis=1)
# iqr1 = df1.apply(out_iqr, k=1.5)
# iqr1.head(10)
# 
# for column in df1:
#     df1[column] = np.where(iqr1[column] == True,'NaN',df1[column])
# 
# cols = df1.columns
# df1[cols] = df1[cols].apply(pd.to_numeric, errors='coerce')
# df1.head(10)
# 
# #Use linear interpolation to fill up nulls
# clean_data = df1.interpolate(method='linear', axis=0).bfill().ffill()
# df['Date'] = pd.to_datetime(df['Date'])
# clean_data = pd.concat([df['Date'],clean_data], axis=1)
# clean_data.head(10)
# 
# =============================================================================












