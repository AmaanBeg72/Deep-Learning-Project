#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[6]:


df=pd.read_csv(r"C:\Users\amaan\Downloads\creditcard.csv")


# In[7]:


df


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


df.drop(['Time'],axis=1,inplace=True)


# In[13]:


df.head()


# In[16]:


df.shape


# In[17]:


x=df.iloc[:,:-1]


# In[18]:


x


# In[19]:


y=df.iloc[:,-1]


# In[20]:


y


# In[22]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=pd.DataFrame(sc.fit_transform(x),columns=x.columns)


# In[23]:


x


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,mean_squared_error,accuracy_score


# In[25]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=1)


# In[26]:


xtest.shape


# In[27]:


xtrain.shape


# In[28]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[37]:


model=Sequential()
model.add(Dense(16,activation="relu",input_dim=29))
model.add(Dense(8,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(1,activation="sigmoid"))


# In[38]:


model.compile(optimizer="adam",loss="binary_crossentropy",metrics="accuracy")


# In[39]:


his=model.fit(xtrain,ytrain,epochs=50,batch_size=100)


# In[40]:


his.history["loss"]


# In[41]:


ypred=model.predict(xtest)


# In[42]:


ypred


# In[44]:


ypred=np.where(ypred>=0.5,1,0)
ypred


# In[45]:


print(classification_report(ytest,ypred))


# In[47]:


print(accuracy_score(ytest,ypred))

