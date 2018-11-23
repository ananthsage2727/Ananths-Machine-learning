
# coding: utf-8

# In[5]:

import pandas as pd
import matplotlib.pyplot as plt   
weatherhistory = pd.read_csv("D:\downloadss\weatherHistory.csv")
weatherhistory.head()
weatherhistory.info()
weatherhistory.describe()
weatherhistory.columns


# In[8]:

weatherhistory.head(10)


# In[17]:

X = weatherhistory[['Temperature (C)']]
y = weatherhistory[['Apparent Temperature (C)']]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
plt.scatter(y_test,lm.predict(X_test))
plt.show()


# In[ ]:



