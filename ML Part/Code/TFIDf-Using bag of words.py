#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
metadata = pd.read_csv('c10.csv')
#df.head()


# In[2]:


#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
metadata['Bag_of_words'] = metadata['Bag_of_words'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['Bag_of_words'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# In[3]:


tfidf.get_feature_names()[7000:7010]


# In[4]:


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[5]:


cosine_sim.shape


# In[6]:


cosine_sim[1]


# In[7]:


#Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['0']).drop_duplicates()


# In[8]:


indices[:10]


# In[9]:


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['0'].iloc[movie_indices]


# In[10]:


get_recommendations('/bg/6/45/')


# In[11]:


i=1
metadata['tfidf']=''
for index, row in metadata.iterrows():
    row['tfidf']=get_recommendations(row['0']).to_list()
    print(row['0'])
    print(get_recommendations(row['0']).to_list())
    print(i)
    i=i+1


# In[13]:


c=metadata.to_csv("ct10.csv",index=False)


# In[ ]:




