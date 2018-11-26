
# coding: utf-8

# In[8]:

import numpy as np
import pandas as pd
from sklearn import tree

input_file = "D:\downloadss/Iris.csv"
df = pd.read_csv(input_file, header = 0)


# In[4]:

df.head(76
    )


# In[9]:

d = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df['Species'] = df['Species'].map(d)
df.head(149)


# In[10]:

features = list(df.columns[:5])
features


# In[11]:

y = df["Species"]
X = df[features]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)


# In[12]:

from IPython.display import Image  
from sklearn.externals.six import StringIO  
import pydotplus

dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=features)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  


# In[13]:

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, y)


# In[24]:


print (clf.predict([[4, 3, 4, 4, 4, ]]))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



