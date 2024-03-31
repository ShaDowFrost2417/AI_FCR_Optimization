#!/usr/bin/env python
# coding: utf-8

# # Process the many clusters to 1 file

# ## Import Functions

# In[1]:


from AnalysisTools.postprocessing_utils import *
import pandas as pd


# ## Processing

# In[ ]:


save_path = 'D:\DETR\parquets\processed'


# ### Twice a day data

# In[ ]:


file_path = 'D:\DETR\parquets\Twice_a_Day'
df = combine_df_from_files(file_path)


# In[ ]:


df = generate_parameter_values(df,2)


# In[ ]:


# save processed data
df.to_parquet(save_path + '\twice_a_day_complete_processed.parquet')


# In[ ]:


df_graphing = generate_graphing_df(df)
df_graphing.to_parquet(save_path + '\twice_a_day_5minMovingAverage_Graphing.parquet')


# In[ ]:


df_clustering = generate_clustering_df(df)
df_clustering.to_parquet(save_path + '\twice_a_day_5minMovingAverage_Clustering.parquet')


# ### Once a day data

# In[ ]:


file_path = 'D:\DETR\parquets\Once_a_Day'
df = combine_df_from_files(file_path)


# In[ ]:


df = generate_parameter_values(df,1)


# In[ ]:


# save processed data
df.to_parquet(save_path + '\once_a_day_complete_processed.parquet')


# In[ ]:


df_graphing = generate_graphing_df(df)
df_graphing.to_parquet(save_path + '\once_a_day_5minMovingAverage_Graphing.parquet')

