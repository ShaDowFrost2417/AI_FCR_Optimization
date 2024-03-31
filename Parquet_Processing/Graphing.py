#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


from AnalysisTools.graphing_utils import *
import pandas as pd


# # Read Files

# In[7]:


file_path = 'D:/DETR/parquets/processed'

#Twice a day
df_graphing_twiceaday = pd.read_parquet(file_path + '/twice_a_day_5minMovingAverage_Graphing.parquet')
# Convert the index (TimeNumeric in seconds) to hours
df_graphing_twiceaday.index = df_graphing_twiceaday.index / 3600  # Converts index to hours
df_graphing_twiceaday["Feeding_Freq"] = 2


df_graphing_onceaday = pd.read_parquet(file_path + '/once_a_day_5minMovingAverage_Graphing.parquet')
# Convert the index (TimeNumeric in seconds) to hours
df_graphing_onceaday.index = df_graphing_onceaday.index / 3600  # Converts index to hours
df_graphing_onceaday["Feeding_Freq"] = 1


# In[7]:


df_graphing = pd.concat([df_graphing_twiceaday, df_graphing_onceaday])


# # Plotting Graphs

# In[18]:


count1, count2 = 0, 0
for column in df_graphing.columns.drop(["Date","Feeding_Freq"]):
    plot_1_stdev_band(df_graphing_twiceaday, column)
    
    count2 += 1
    plot_1_stdev_band(df_graphing_onceaday, column)
    
    count2 += 1
    plot_1_stdev_band(df_graphing, column)
    
    count1 += 1
    count2 = 0


# In[8]:


plot_3D_1_stdev_band(df_graphing, 'Weighted_Position_x', 'Weighted_Position_y')


# In[9]:


plot_3D_1_stdev_band(df_graphing, 'Pos_Var_x', 'Pos_Var_y')

