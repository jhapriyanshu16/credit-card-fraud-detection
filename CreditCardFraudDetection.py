#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('creditcard.csv')


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])


# In[7]:


data.info()


# In[8]:


#check null values: 
data.isnull().sum()


# In[9]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler


# In[10]:


sc = StandardScaler()
data['Amount']=sc.fit_transform(pd.DataFrame(data['Amount']))


# In[11]:


data.head()


# In[12]:


data = data.drop(['Time'],axis=1)


# In[13]:


data.head()


# In[14]:


data.shape


# In[15]:


data.duplicated().any()


# In[16]:


data = data.drop_duplicates()


# In[17]:


data.shape


# In[18]:


284807- 275663


# In[19]:


#Not Handling Imbalanced
data['Class'].value_counts()


# In[21]:


import matplotlib.pyplot as plt

# Assuming 'Class' is a categorical variable in your DataFrame 'data'
plt.figure(figsize=(8, 6))
data['Class'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Count Plot of Class')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# In[22]:


#Store Feature Matrix In X And Response (Target) In Vector y
X = data.drop('Class',axis=1)
y = data['Class']


# In[23]:


#Splitting The Dataset Into The Training Set And Test Set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,
                                                 random_state=42)


# In[24]:


# Handling Imbalanced Dataset
# Undersampling
normal = data[data['Class']==0]
fraud = data[data['Class']==1]


# In[25]:


normal.shape


# In[26]:


fraud.shape


# In[27]:


normal_sample=normal.sample(n=473)


# In[28]:


normal_sample.shape


# In[29]:


new_data = pd.concat([normal_sample,fraud],ignore_index=True)


# In[30]:


new_data['Class'].value_counts()


# In[31]:


new_data.head()


# In[32]:


X = new_data.drop('Class',axis=1)
y = new_data['Class']


# In[33]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,
                                                 random_state=42)


# In[34]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)


# In[35]:


y_pred1 = log.predict(X_test)


# In[36]:


from sklearn.metrics import accuracy_score


# In[37]:


accuracy_score(y_test,y_pred1)


# In[38]:


from sklearn.metrics import precision_score,recall_score,f1_score


# In[39]:


precision_score(y_test,y_pred1)


# In[40]:


recall_score(y_test,y_pred1)


# In[41]:


f1_score(y_test,y_pred1)


# In[42]:


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[43]:


y_pred2 = dt.predict(X_test)


# In[44]:


accuracy_score(y_test,y_pred2)


# In[45]:


precision_score(y_test,y_pred2)


# In[46]:


recall_score(y_test,y_pred2)


# In[47]:


f1_score(y_test,y_pred2)


# In[48]:


#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)


# In[49]:


y_pred3 = rf.predict(X_test)


# In[50]:


accuracy_score(y_test,y_pred3)


# In[51]:


precision_score(y_test,y_pred3)


# In[52]:


recall_score(y_test,y_pred3)


# In[53]:


f1_score(y_test,y_pred3)


# In[54]:


final_data = pd.DataFrame({'Models':['LR','DT','RF'],
              "ACC":[accuracy_score(y_test,y_pred1)*100,
                     accuracy_score(y_test,y_pred2)*100,
                     accuracy_score(y_test,y_pred3)*100
                    ]})


# In[55]:


final_data


# In[56]:


import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))
plt.bar(final_data['Models'], final_data['ACC'], color='lightblue')
plt.xlabel('Models')
plt.ylabel('ACC')
plt.title('Bar Plot of ACC for Each Model')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Adjust layout for better appearance
plt.show()


# In[57]:


# Oversampling
X = data.drop('Class',axis=1)
y = data['Class']


# In[58]:


X.shape


# In[59]:


y.shape


# In[60]:


from imblearn.over_sampling import SMOTE


# In[61]:


X_res,y_res = SMOTE().fit_resample(X,y)


# In[62]:


y_res.value_counts()


# In[63]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size=0.20,
                                                 random_state=42)


# In[64]:


# Logistic Regression
log = LogisticRegression()
log.fit(X_train,y_train)


# In[65]:


y_pred1 = log.predict(X_test)


# In[66]:


accuracy_score(y_test,y_pred1)


# In[67]:


precision_score(y_test,y_pred1)


# In[68]:


recall_score(y_test,y_pred1)


# In[69]:


f1_score(y_test,y_pred1)


# In[70]:


# Decision Tree Classifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[71]:


y_pred2 = dt.predict(X_test)


# In[72]:


accuracy_score(y_test,y_pred2)


# In[73]:


precision_score(y_test,y_pred2)


# In[74]:


recall_score(y_test,y_pred2)


# In[75]:


f1_score(y_test,y_pred2)


# In[76]:


# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)


# In[77]:


y_pred3 = rf.predict(X_test)


# In[78]:


accuracy_score(y_test,y_pred3)


# In[79]:


precision_score(y_test,y_pred3)


# In[80]:


recall_score(y_test,y_pred3)


# In[81]:


f1_score(y_test,y_pred3)


# In[82]:


final_data = pd.DataFrame({'Models':['LR','DT','RF'],
              "ACC":[accuracy_score(y_test,y_pred1)*100,
                     accuracy_score(y_test,y_pred2)*100,
                     accuracy_score(y_test,y_pred3)*100
                    ]})


# In[83]:


final_data


# In[84]:


import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))
plt.bar(final_data['Models'], final_data['ACC'], color='lightblue')
plt.xlabel('Models')
plt.ylabel('ACC')
plt.title('Bar Plot of ACC for Each Model')
plt.xticks(rotation=45, ha='right')  
plt.tight_layout()  # Adjust layout for better appearance
plt.show()


# In[85]:


# Save The Model
rf1 = RandomForestClassifier()
rf1.fit(X_res,y_res)


# In[86]:


import joblib


# In[87]:


joblib.dump(rf1,"credit_card_model")


# In[88]:


model = joblib.load("credit_card_model")


# In[89]:


pred = model.predict([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])


# In[90]:


if pred == 0:
    print("Normal Transcation")
else:
    print("Fraudulent Transcation")

