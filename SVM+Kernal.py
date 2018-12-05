
# coding: utf-8

# In[4]:

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd 


# In[5]:

irisdata = pd.read_csv("D:/downloadss/Iris.csv")


# In[6]:

X = irisdata.drop('Class', axis=1)  
y = irisdata['Class']  


# In[8]:

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  


# In[9]:

from sklearn.svm import SVC  
svclassifier = SVC(kernel='poly', degree=8)  
svclassifier.fit(X_train, y_train)  


# In[10]:

y_pred = svclassifier.predict(X_test)  


# In[11]:

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[ ]:



