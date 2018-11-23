import pandas as pd
import matplotlib.pyplot as plt   #Data visualisation libraries 
movieratings = pd.read_csv("D:/downloadss/tmdb_5000_movies.csv")
movieratings.head()
movieratings.info()
movieratings.describe()
movieratings.columns
X = movieratings[['budget','popularity']]
y = movieratings['revenue']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
fitline = lm.predict(X_test)
plt.scatter(y_test,fitline)
plt.show()