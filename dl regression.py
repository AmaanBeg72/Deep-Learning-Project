#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[5]:


df=pd.read_csv(r"C:\Users\amaan\Downloads\USA_Housing.csv")


# In[6]:


df


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.drop(['Address'],axis=1,inplace=True)


# In[11]:


df.head()


# In[13]:


df.shape


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler


# In[14]:


x=df.iloc[:,:-1]


# In[15]:


x


# In[16]:


y=df.iloc[:,-1]


# In[17]:


y


# In[19]:


sc=StandardScaler()
x=pd.DataFrame(sc.fit_transform(x),columns=x.columns)


# In[20]:


x.head()


# In[21]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[22]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=1)


# In[23]:


xtrain.shape


# In[109]:


model=Sequential()
model.add(Dense(64,activation="relu",input_dim=5))
model.add(Dense(32,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(1))


# In[110]:


model.compile(optimizer="adam",loss="mse")


# In[111]:


his=model.fit(xtrain,ytrain,epochs=50,batch_size=10)


# In[112]:


his.history["loss"]


# In[113]:


ypred=model.predict(xtest)


# In[114]:


ypred


# In[115]:


mean_squared_error(ytest,ypred)


# In[116]:


print(r2_score(ytest,ypred))

