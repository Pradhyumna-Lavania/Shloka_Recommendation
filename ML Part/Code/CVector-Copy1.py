#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install rake-nltk')


# In[3]:


from rake_nltk import Rake
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# In[4]:


import pandas as pd
df = pd.read_csv('withrec.csv')


# In[5]:


df['2'] = df['2'].map(lambda x: x.split(' '))
df['5'] = df['5'].map(lambda x: x.split(' '))


# In[6]:


for index, row in df.iterrows():
        row['2'] = [x.lower() for x in row['2']]
        row['5'] = [x.lower().replace(',','').replace(':','') for x in row['5']]
#df.head()


# In[7]:


df1=pd.read_csv("common.csv")


# In[8]:


l2=df1['W1'].to_list()
l1=df1['W2'].to_list()
l=l1+l2+['are']
#print (l)
#print(len(l))


# In[9]:


for index, row in df.iterrows():
    m=row['5']
    for x in (m):
        if ((x in l)==True):
            m.remove(x)
    
       


# In[10]:


df['Bag_of_words'] = ''
columns = ['2','5']
for index, row in df.iterrows():
    words = ''
    for col in columns:
        words += ' '.join(row[col]) + ' '
    row['Bag_of_words'] = words
    #print(row['Bag_of_words'])
    
df = df[['0','1','2','3','4','5','6','7','8','9','Bag_of_words']]


# In[14]:


len(df['0'].to_list())


# In[29]:


count = CountVectorizer()
count_matrix = count.fit_transform(df['Bag_of_words'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
mat=cosine_sim
print(cosine_sim)


# In[30]:


indices = pd.Series(df['0'])


# In[31]:


def recommend(title, cosine_sim = cosine_sim):
    recommended_shlokas = []
    idx = indices[indices == title].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_10_indices = list(score_series.iloc[1:10].index)
    
    for i in top_10_indices:
        recommended_shlokas.append(list(df['0'])[i])
        
    return recommended_shlokas


# In[32]:


(recommend('/bg/3/10/'))


# In[15]:


df['10']=''
for index, row in df.iterrows():
    row['10']=recommend(row['0'])
    #print(recommend(row['0']))


# In[ ]:


df['cvscore']=''
for index, row in df.iterrows():
    i=df['0'].index(row['0'])
    j=[]
    for d in row['10']:
        j.append(m[i][df['0'].index(d)])
    


# In[33]:


l=['a','b','c']
print(l.index('b'))


# In[16]:


a=df.to_csv("cv.csv",index=False)


# In[ ]:




