import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv('../chap3/data/iris.data', names=names)


# In[3]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

print(X)
print(y)