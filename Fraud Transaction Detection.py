#!/usr/bin/env python
# coding: utf-8

# # Creating a model to detect Fraudulent transactions

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


fraud = pd.read_csv(r'C:\Users\saatv\Downloads\Fraud.csv')


# In[3]:


fraud.head()


# In[4]:


fraud.describe()


# In[5]:


fraud.corr()


# In[6]:


fraud.info()


# In[7]:


## Features and target creations
X = fraud.drop(['isFraud'], axis=1)
y = fraud[['isFraud']]


# In[8]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['type'] = le.fit_transform(X.type)


# In[9]:


X=X.drop('nameOrig', axis=1)
X=X.drop('nameDest', axis=1)


# In[10]:


X.shape


# In[11]:


y.shape


# In[12]:


X.head(10)


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[14]:


X_train.shape


# In[15]:


## Building decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[16]:


def decision_tree_classification(X_train, y_train, X_test, y_test):
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train.values.ravel())
    score = classifier.score(X_test, y_test)
    y_pred = classifier.predict(X_test)
    print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report :- \n {classification_report(y_test, y_pred)}")
    
decision_tree_classification(X_train, y_train, X_test, y_test)


# 1. There are no missing values in the dataset.This was checked by using dropna() function from Pandas.
#    To check for outliers, we take the Inter-Quartile Range * 1.5 and check for values than this in the dataset.

# 2. My Fraudulent Transaction detector uses a Decision Tree classifier to predict Fraudulent Transactions. Considering the result, if the result is 1 then the transaction is fraudulent, if it's 0, then it's not.
#    I have used Python Pandas for data preprocessing and sci-kit learn to develop the classifier.

# 3. The dataset was split into X and y as attributes and labels respectively.
#    I used the generic terms for splitting the dataset into training and testing sets i.e. X_train, X_test, y_train, y_test    and all other variables are used to develop the classifier for the problem.

# 4. I used the confusion_matrix and classification_report class from sklearn to find the precision, recall and f1 score of the model.
#    The average precision comes out to be 0.89.

# 5. I used the corr() function from pandas to find the correlation between different feature as we can see above in the        model.The amount of transaction and the type of transaction are the key features affecting the prediction of fraudulent    transaction with the highest and the second highest correlation coefficients respectively

# 6. These factors do make sense, as given in the dictionary fo the dataset, the flagged fraudulent transactions are the ones    which are more than an amount of 200.00. 

# 7. The customer's profile must be checked carefully for previous transactions and different scores related to their            financial transactions. The amount and the number of transactions occuring frequently must also be taken care of.

# 8. Using the same model we can determine if the transactions are fraudulent or not.

# In[ ]:




