
# coding: utf-8

# In[1]:

import numpy as np  
import pandas as pd  


# In[2]:

dataset = pd.read_csv("D:/downloadss/Datasets/Iris.csv")  


# In[3]:

dataset.head()  


# In[4]:

X = dataset.drop('Class', 1)  
y = dataset['Class'] 


# In[5]:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


# In[6]:

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  


# In[7]:

from sklearn.decomposition import PCA

pca = PCA()  
X_train = pca.fit_transform(X_train)  
X_test = pca.transform(X_test)  


# In[8]:

explained_variance = pca.explained_variance_ratio_  


# In[21]:

from sklearn.decomposition import PCA

pca = PCA(n_components=3)  
X_train = pca.fit_transform(X_train)  
X_test = pca.transform(X_test)  


# In[10]:

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)  
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[12]:

from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)  
print(cm)  


# In[17]:

print( accuracy_score(y_test, y_pred))


# In[ ]:



