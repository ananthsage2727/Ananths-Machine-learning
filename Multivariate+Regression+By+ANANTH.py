
# coding: utf-8

# In[2]:

import pandas as pd
df = pd.read_csv('D:/downloadss/tmdb_5000_movies.csv')
df.head()


# In[8]:

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
import matplotlib.pyplot as plt

X = df[['vote_count', 'popularity']]
y = df['revenue']

X[['vote_count', 'popularity']] = scale.fit_transform(X[['vote_count', 'popularity']].as_matrix())

print (X)

est = sm.OLS(y, X).fit()

est.summary()


# In[7]:

plt.plot(y,X)
plt.show()


# In[ ]:



