#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import pytz
data = pd.read_csv("Real/surveys.csv")
users = ['Z23ATY', 'C09MBB', 'S13NBG']
data = data[(data['code'].isin(users)) ]
data = data.replace(users, ['P1', 'P2', 'P3'])


# In[3]:


data.code.unique()


# In[4]:


utc_times = pd.to_datetime(data['time_issued'], unit='ms').dt.tz_localize(tz=pytz.utc)
aus_times = utc_times.dt.tz_convert(tz=pytz.timezone('Australia/Melbourne'))
aus_date = aus_times.dt.date
data['date'] = aus_date


# In[5]:


before = data[data['time_issued'] < 1583500000000]
during = data[data['time_issued'] > 1583500000000]
before.columns


# In[6]:


before['esm_role'].value_counts()


# In[7]:


def get_workday_sizes(df):
    return df.groupby(['code', 'date'])['esm_role'].value_counts().unstack().fillna(0).apply(lambda x :  (x[2]) / (x[1] + x[2] + x[3]), axis=1)
def get_workday_size(df):
    return get_workday_sizes(df).groupby(level=0).mean()
full = pd.DataFrame({'before': get_workday_size(before), 'during': get_workday_size(during)})
full
full.plot.bar()


# In[8]:


full = pd.DataFrame({'before': get_workday_sizes(before), 'during': get_workday_sizes(during)}).rename_axis(index={'code': 'Participant'})
full


# In[9]:


full.columns


# In[10]:


import scipy.stats as stats
def ttest(df):
    result = stats.ttest_ind(df['before'], df['during'], nan_policy='omit')
    return pd.Series({'statistic': result[0], 'pvalue': result[1]})
full.groupby(level=0).apply(ttest).columns


# In[21]:


import matplotlib.pyplot as plt

full.boxplot(by='Participant', labels=['Participant 1', 'Participant 2', 'Participant 3'], figsize=(5,5))


# In[12]:


full


# In[13]:


data.groupby(['code', data['time_issued'] > 1583500000000])['esm_role'].value_counts()


# In[14]:


line_df = pd.DataFrame({'work proportions': pd.concat([get_workday_sizes(before),get_workday_sizes(during)])}).reset_index()
line_df


# In[15]:


line_df.plot.scatter(x='date', y='work proportions')


# In[16]:


df = data.groupby(['code', 'date', data['time_issued'] > 1583500000000])['esm_valence'].mean().unstack()
df


# In[17]:


df.rename(columns={False: 'before', True: 'during'})


# In[ ]:




