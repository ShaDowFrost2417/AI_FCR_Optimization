#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[ ]:


import pandas as pd
import numpy as np
import os
# import datetime
# import time


# # Function Definitions

# In[ ]:


def combine_df_from_files(file_path):
    df_list = []
    
    for file in os.listdir(file_path):
        if file.split('.')[-1] == 'parquet':
            file_directory = os.path.join(file_path, file)
            df_list.append(pd.read_parquet(file_directory))
    
    df = pd.concat(df_list, ignore_index=True)
    return df


# In[ ]:


def checking_corrupt_files(file_path):
    corrupt_files = []

    for file in os.listdir(file_path):
        file_directory = os.path.join(file_path, file)
        try:
            test_df = pd.read_parquet(file_directory)
        except:
            corrupt_files.append(file)

    print(test_df)


# In[ ]:


def generate_parameter_values(df, feeding_frequency_per_day):
    #Add any other parameters here in the future if needed!
    
    df['Weighted_Position_x'] = df['COG'].apply(lambda x: x[0])
    df['Weighted_Position_y'] = df['COG'].apply(lambda x: x[1])

    # square this cos actually inside is still stdev. stdev square is variance.
    df['Pos_Var_x'] = df['Pos_Var_x'] ** 2
    df['Pos_Var_y'] = df['Pos_Var_y'] ** 2

    # Convert datetime.time to minutes since midnight
    # TimeNumeric is in seconds while Time_After_Feeding is in minutes
    df['TimeNumeric'] = df['Time'].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second + t.microsecond * (10**(-6)))
    
    interval_between_feeding_seconds = 86400/feeding_frequency_per_day
    df["Time_After_Feeding"] = df["TimeNumeric"].apply(lambda x: ((x-28800)%interval_between_feeding_seconds)/60)
    df["Avg_Dir_shifted"] = df["Avg_Dir"].shift(1)
    df["Turning_Angle"] = np.minimum(abs(df["Avg_Dir"] - df["Avg_Dir_shifted"]) % 180, abs(df["Avg_Dir_shifted"] - df["Avg_Dir"]) % 180)
    
    return df


# In[ ]:


def generate_graphing_df(df):
    # Take relevant data for graphing
    df_graphing = df[['Pos_Var_x', 'Pos_Var_y','Avg_Speed', 'Lingering_Count', 'Avg_Dir', 'Weighted_Position_x', 'Weighted_Position_y', 'Turning_Angle']]

    # Get 5-minute moving averages of the data
    # Calculate the rolling window size
    window_size = 5 * 60 * 20  # 5 minutes * 60 seconds * 20 data points per second
    # Note: The result's first few rows up to window_size-1 will be NaN because there's not enough data to calculate the moving average
    df_graphing = df_graphing.rolling(window=window_size, min_periods=20).mean()
    df_graphing.dropna(inplace = True)

    # Match the indexes with the original df to get the Date and TimeNumeric
    df_graphing['Date'] = df.loc[df_graphing.index, 'Date']
    df_graphing['TimeNumeric'] = df.loc[df_graphing.index, 'TimeNumeric']

    # Set 'TimeNumeric' as index
    df_graphing.set_index('TimeNumeric', inplace=True)
    
    return df_graphing


# In[ ]:


def generate_clustering_df(df):
    #Take relevant data for clustering, set time after feeding as index
    df_clustering = df[['Time_After_Feeding', 'Pos_Var_x', 'Pos_Var_y','Avg_Speed', 'Lingering_Count', 'Avg_Dir', 'Weighted_Position_x', 'Weighted_Position_y', 'Turning_Angle']]
    df_clustering.set_index('Time_After_Feeding', inplace=True)

    #Get 5-minute moving averages of the data
    # Calculate the rolling window size
    window_size = 5 * 60 * 20  # 5 minutes * 60 seconds * 20 data points per second
    # Note: The result's first few rows up to window_size-1 will be NaN because there's not enough data to calculate the moving average
    df_clustering = df_clustering.rolling(window=window_size, min_periods=20).mean()
    df_clustering.dropna(inplace = True)

    return df_clustering

