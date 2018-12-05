
# coding: utf-8

# In[1]:

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().magic('matplotlib inline')


# In[2]:

dataset = pd.read_csv("D:/downloadss/bill_authentication.csv")  


# In[3]:

X = dataset.drop('Class', axis=1)  
y = dataset['Class'] 


# In[4]:

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  


# In[5]:

from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, y_train)  


# In[6]:

y_pred = classifier.predict(X_test)  


# In[7]:

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[ ]:



