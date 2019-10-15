#!/usr/bin/env python
# coding: utf-8

# In[6]:


# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from plotly.offline  import download_plotlyjs,init_notebook_mode,plot, iplot
import cufflinks as cf
init_notebook_mode(connected = True)
cf.go_offline()
get_ipython().run_line_magic('matplotlib', 'inline')

from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.offline as offline
# Squarify for treemaps
import squarify
# Random for well, random stuff
import random
# operator for sorting dictionaries
import operator
# For ignoring warnings
import warnings
warnings.filterwarnings('ignore')



#df = pd.read_csv("F:/Sem II/Data Visyalization and Story telling/data.csv")
#df.head(2)


# In[7]:


df = pd.read_csv("F:/Sem II/Data Visyalization and Story telling/data.csv")
df.tail(2)


# In[133]:


df.info()


# In[6]:


df.describe()


# In[130]:


df.describe(include="all")


# In[12]:



print("Number of records : ", df.shape[0])
print("Number of Blue fighters : ", len(df.B_fighter.unique()))
print("Number of Red fighters : ", len(df.R_fighter.unique()))


# In[1]:



df['B_age'] = df['B_age'].fillna(np.mean(df['B_age']))
df['B_Height_cms'] = df['B_Height_cms'].fillna(np.mean(df['B_Height_cms']))
df['R_age'] = df['R_age'].fillna(np.mean(df['R_age']))
df['R_Height_cms'] = df['R_Height_cms'].fillna(np.mean(df['R_Height_cms']))


# In[2]:


df.isnull().sum(axis=0)


# # <a> Data Visualization</a>
# 
# Let's start by looking who's winning more from our dataset:
# 
# 

# In[12]:


temp = df["Winner"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, 1]},
      "hole": .6,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Winner",
        "annotations": [
            {
                "font": {
                    "size": 17
                },
                "showarrow": False,
                "text": "Whos winning?",
                "x": 0.5,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')


# In[22]:


fig, ax = plt.subplots(1,2, figsize=(15, 5))
sns.distplot(df.B_age, ax=ax[0])
sns.distplot(df.R_age, ax=ax[1])


# In[14]:


BAge = df.groupby(['B_age']).count()['Winner']
BlueAge = BAge.sort_values(axis=0, ascending=False)
blue = BlueAge.head(10)
BlueAge.head(10)


# In[2]:


sns.distplot(BlueAge.head(10), ax=ax[0])


# In[16]:


RAge = df.groupby(['R_age']).count()['Winner']
RedAge = RAge.sort_values(axis=0, ascending=False)
RedAge.head(10)


# In[17]:


RAge = df.groupby(['R_age']).count()['Winner']
RedAge = RAge.sort_values(axis=0, ascending=False)
RedAge = RedAge.head(10)

BAge = df.groupby(['B_age']).count()['Winner']
BlueAge = BAge.sort_values(axis=0, ascending=False)
BlueAge = BlueAge.head(10)

figR = {
  "data": [
    {
      "values": RedAge.values,
      "labels": RedAge.index,
      "domain": {"x": [0, 1]},
      "hole": .6,
      "type": "pie"
    },
    
    ],
  "layout": {
        #"title":"Challenger",
        "annotations": [
            {
                "font": {
                    "size": 17
                },
                "showarrow": False,
                "text": "Challengers Age",
                "x": 0.5,
                "y": 0.5
            }
            
        ]
    }
}

figB = {
  "data": [
    {
      "values": BlueAge.values,
      "labels": BlueAge.index,
      "domain": {"x": [0, 1]},
      "hole": .6,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Champion",
        "annotations": [
            {
                "font": {
                    "size": 17
                },
                "showarrow": False,
                "text": "Champions Age",
                "x": 0.5,
                "y": 0.5
            }
            
        ]
    }
}
iplot(figR, filename='donut')
iplot(figB, filename='donut')


# In[3]:


fig, ax = plt.subplots(1,2, figsize=(15, 5))
above35 =['above35' if i >= 35 else 'below35' for i in df.B_age]
df_B = pd.DataFrame({'Champion':above35})
sns.countplot(x=df_B.B_age, ax=ax[0])
plt.ylabel('Number of fighters')
plt.title('Age of Blue fighters',color = 'blue',fontsize=15)

above35 =['above35' if i >= 35 else 'below35' for i in df.R_age]
df_R = pd.DataFrame({'Challenger':above35})
sns.countplot(x=df_R.R_age, ax=ax[1])
plt.ylabel('Number of Red fighters')
plt.title('Age of Red fighters',color = 'Red',fontsize=15)


# 

# In[30]:


df['Age_Difference'] = df.B_age - df.R_age
df[['Age_Difference', 'Winner']].groupby('Winner').mean()


# 

# In[25]:


fig, ax = plt.subplots(1,2, figsize=(15, 5))
sns.distplot(df.R_Height_cms, bins = 20, ax=ax[0]) #Blue 
sns.distplot(df.B_Height_cms, bins = 20, ax=ax[1]) #Red


# In[34]:


fig, ax = plt.subplots(figsize=(14, 6))
sns.kdeplot(df.B_Height_cms, shade=True, color='indianred', label='Red')
sns.kdeplot(df.R_Height_cms, shade=True, label='Blue')


# In[36]:


df['Height_Difference'] = df.B_Height_cms - df.R_Height_cms
df[['Height_Difference', 'Winner']].groupby('Winner').mean()


# In[27]:


win = pd.read_csv("F:/Sem II/Data Visyalization and Story telling/UFC/UFCdatavisualization.csv")


# 

# In[150]:


temp = win["winby"].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, 1]},
      #"name": "Types of Loans",
      #"hoverinfo":"label+percent+name",
      "hole": .6,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"How the fighter's are winning?",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Win by",
                "x": 0.50,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')


# 

# In[152]:


g = sns.FacetGrid(win, col='winby')
g.map(plt.hist, 'R_Age', bins=50)


# In[28]:


g = sns.FacetGrid(win, col='winby')
g.map(plt.hist, 'R_Age', bins=20)


# In[29]:


g = sns.FacetGrid(win, col='winby')
g.map(plt.hist, 'B_Age', bins=20)


# 

# In[159]:


g = sns.FacetGrid(df, col='weight_class')
g.map(plt.hist, 'R_age', bins=5)


# In[46]:


sns.lmplot(x="B_avg_BODY_att", 
               y="B_avg_BODY_landed", 
               col="Winner", hue="Winner", data=df, col_wrap=2, size=6)


# In[ ]:





# Attempts and strikes landed are, as expected, perfectly linear.
# 
# Now, let's look at the location and find out most popular countries

# In[88]:


cnt_srs = df['R_Location'].value_counts().head(15)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
    ),
)

layout = go.Layout(
    title='Most Popular cities for Red fighters'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Ratio")


# In[135]:


cnt_srs = df['location'].value_counts().head(15)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
    ),
)

layout = go.Layout(
    title='Most Popular cities for fighters'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Ratio")


# In[86]:


cnt_srs = df['B_Location'].value_counts().head(15)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
    ),
)

layout = go.Layout(
    title='Most Popular cities for Blue fighters'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Ratio")


# MMA seems to be most prominent in Brazil and USA. Infact, MMA is second most popular sport after Soccer in Brazil. I wonder if it is due to ancient Brazilian Jiu-Jitsu?
# 
# Now, let's look at the Grappling reversals, grappling standups and grappling takedowns landed in different weight categories in** Round 1**

# In[160]:



r1 = df[['B_Weight_lbs', 'B_avg_BODY_landed', 'B_avg_CLINCH_landed', 'B_avg_GROUND_landed']].groupby('B_Weight_lbs').sum()

r1.plot(kind='line', figsize=(10,10))
plt.show()


# In[32]:


inputs = pd.read_csv("F:/Sem II/Data Visyalization and Story telling/UFC/UFCdatavisualization.csv")


# In[33]:



r5 = inputs[['B_Weight', 'B__Round1_Grappling_Reversals_Landed', 'B__Round1_Grappling_Standups_Landed', 'B__Round1_Grappling_Takedowns_Landed']].groupby('B_Weight').sum()

r5.plot(kind='line', figsize=(14,6))
plt.show()


# There are very few Grappling reversals but high amount of Grappling takedowns that were landed. More specifically weight classes between 70 - 80 prefer takedowns during Round 1. 
# 
# Let's compare the same for Round 5

# In[113]:



r5 = inputs[['B_Weight', 'B__Round5_Grappling_Reversals_Landed', 'B__Round5_Grappling_Standups_Landed', 'B__Round5_Grappling_Takedowns_Landed']].groupby('B_Weight').sum()

r5.plot(kind='line', figsize=(14,6))
plt.show()


# Interestingly, grappling reversals increase for fighters between weight 80-90, while takedowns have decreased in the lighter weight groups.
# 
# Lets look similar data for Clinch head strikes, Clinch leg strikes and Body strikes for Round 1

# In[114]:


clin_r1 = inputs[['B_Weight', 'B__Round1_Strikes_Clinch Head Strikes_Landed', 'B__Round1_Strikes_Clinch Leg Strikes_Landed', 'B__Round1_Strikes_Clinch Body Strikes_Landed']].groupby('B_Weight').sum()

clin_r1.plot(kind='line', figsize=(14,6))
plt.show()


# Fighters prefer to land  more head strikes during round 1, let's compare this with what happens in Round 5:

# In[126]:


clin_r5= inputs[['B_Weight', 'B__Round1_Strikes_Clinch Head Strikes_Landed', 'B__Round1_Strikes_Clinch Leg Strikes_Landed', 'B__Round5_Strikes_Clinch Leg Strikes_Landed', 'B__Round5_Strikes_Clinch Head Strikes_Landed']].groupby('B_Weight').sum()

clin_r5.plot(kind='line', figsize=(14,6))
plt.show()


# In[120]:


clin= inputs[['B_Weight', 'B__Round1_Strikes_Clinch Head Strikes_Landed', 'B__Round2_Strikes_Clinch Head Strikes_Landed', 'B__Round3_Strikes_Clinch Head Strikes_Landed', 'B__Round4_Strikes_Clinch Head Strikes_Landed', 'B__Round5_Strikes_Clinch Head Strikes_Landed']].groupby('B_Weight').sum()

clin.plot(kind='line', figsize=(14,6))
plt.show()


# In[34]:


salaries = pd.read_csv("F:/Sem II/Data Visyalization and Story telling/UFC/salaries.csv")


# 

# In[163]:


clin= salaries[['Name', 'Salaries in 2019']].groupby('Salaries in 2019').sum()

clin.plot(kind='line', figsize=(14,6))
plt.show()


# In[ ]:




