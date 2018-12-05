
# coding: utf-8

# In[1]:

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().magic('matplotlib inline')


# In[2]:

dataset = pd.read_csv('D:/downloadss/petrol_consumption.csv')  


# In[3]:

X = dataset.drop('Petrol_Consumption', axis=1)  
y = dataset['Petrol_Consumption']  


# In[4]:

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  


# In[5]:

from sklearn.tree import DecisionTreeRegressor  
regressor = DecisionTreeRegressor()  
regressor.fit(X_train, y_train)  


# In[6]:

y_pred = regressor.predict(X_test)  


# In[7]:

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  
df  


# In[8]:

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  


# In[ ]:



