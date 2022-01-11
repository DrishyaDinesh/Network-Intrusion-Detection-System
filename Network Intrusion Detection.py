#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

df=pd.read_csv("../input/network-intrusion-detection/Train_data.csv")
df.head()


# In[9]:


missing_data=df.isnull()
for col in missing_data.columns.values.tolist():
    print(col)
    print(missing_data[col].value_counts())
    print("")
#shows there are no missing values


# In[ ]:


df.dtypes


# In[ ]:


df['protocol_type'].value_counts()


# In[ ]:


new1=pd.get_dummies(df["protocol_type"])
df=pd.concat([df,new1],axis=1)
df.drop("protocol_type",axis=1,inplace=True)
df


# In[ ]:


new2=pd.get_dummies(df["service"])
df=pd.concat([df,new2],axis=1)
df.drop("service",axis=1,inplace=True)
df


# In[ ]:


new3=pd.get_dummies(df["flag"])
df=pd.concat([df,new3],axis=1)
df.drop("flag",axis=1,inplace=True)
df


# In[ ]:


df.replace('anomaly',1,inplace=True)
df.replace('normal',0,inplace=True)
df['class']


# In[2]:


features=df.drop(['class'],axis=1)
classlabel=df['class']
X_train, X_test, y_train, y_test = train_test_split(features,classlabel,test_size = 0.2,random_state=1)


# In[3]:


dct=DecisionTreeClassifier()
dct.fit(X_train,y_train)
ypred=dct.predict(X_test)


# In[4]:


acc=accuracy_score(y_test,ypred)
print("Accuracy is: {}%".format(round(acc*100,2)))


# In[5]:


cm=confusion_matrix(y_test,ypred)
fig, ax = plt.subplots(figsize=(3, 3))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=14)
plt.ylabel('Actuals', fontsize=14)
plt.title('Confusion Matrix', fontsize=17)
plt.show()


# In[ ]:





# In[ ]:




