#!/usr/bin/env python
# coding: utf-8

#                                           #Classification Project

# # Import libraries

# In[1]:


import pandas as pd #import pandas library
import numpy as np #import numpy library


# In[2]:


df=pd.read_csv('C:\\Users\\muham\\Downloads\\Machine Learning Project\\Email data set\\emails.csv\\emails.csv') #Read the data using pandas library


# In[3]:


df.shape #check the shape of data


# In[4]:


df.isnull().sum().sum()#check the null values are present or not


# In[5]:


df.head() #Print first rows of data


# In[6]:


df_email_drop=df.drop('Email No.',axis=1)


# # Convert data into X and Y

# In[7]:


X=df_email_drop.drop('Prediction',axis=1) #Drop Prediction columns from data and save to X
Y=df_email_drop['Prediction'] #Get columns of Prediction data
print('Shape of X', X.shape) #Check the shape of Data
print('Shape of Y', Y.shape) #check the shape of Data


# In[8]:


X


# # Split data into train test data

# In[9]:


from sklearn.model_selection import train_test_split #import train test split library from sklearn


# In[10]:


X_train, X_test,Y_train, Y_test=train_test_split(X,Y, test_size=0.2,random_state=51) #Split data into train and test
print('Shape of X_train', X_train.shape) #Check the shape of data
print('Shape of X_test', X_test.shape) #Check the shape of data
print('Shape of Y_train', Y_train.shape) #Check the shape of data
print('Shape of Y_test', Y_test.shape) #Check the shape of data


# # Decision Tree Classifier

# In[11]:


from sklearn.tree import DecisionTreeClassifier  #from sklearn.tree import Decision tree classifier


# In[12]:


DTC=DecisionTreeClassifier(criterion='gini') #create object of Decision tree classifier


# In[13]:


DTC.fit(X_train,Y_train) #train the data using fit 


# # Check the Score of model

# In[14]:


DTC.score(X_test,Y_test)


# # change the criterion and check the score

# In[15]:


DTCE=DecisionTreeClassifier(criterion="entropy") #Run the model model by changing the criterion


# In[16]:


DTCE.fit(X_train, Y_train)#Train the model


# # Check the score of Data

# In[17]:


DTCE.score(X_test,Y_test) #Check the score/accuracy of model


# # Prediction

# In[18]:


X #print the data where I drop prediction columns


# In[19]:


access_index=X.iloc[5171] #using iloc command access rows of corresponding index in one Dimension array
access_index


# In[20]:


email_data_cross_index=np.array([access_index]) #convert one dimension to two dimension data


# In[21]:


pred_value=DTC.predict(email_data_cross_index) #predict the value and save to new variable
pred_value


# # Set the check data are Email spam or not

# In[22]:


if pred_value[0]: #Now implement a check are email spam or not
    print('Spam mail')
else:
    print('Not spam mail')


# # Support Vector Machine Classifiers

# In[23]:


from sklearn.svm import SVC # Import Support vector classifier from sklearn


# In[24]:


svc=SVC(kernel='rbf') #create object of SVC 


# In[25]:


svc.fit(X_train,Y_train) #train the data


# # Check the Accuracy 

# In[26]:


svc.score(X_test,Y_test) #check the score of data


# # Logistic Regression

# In[27]:


from sklearn.linear_model import LogisticRegression #Import logistic Regression from sklearn linear model


# In[28]:


lr=LogisticRegression() #create object of logistic Regression


# In[29]:


lr.fit(X_train,Y_train) #Train the data


# # Check the accuracy level

# In[30]:


lr.score(X_train,Y_train) #check the accuracy level of data


# # Predict the value

# In[31]:


X #just print the data


# In[32]:


index_value=X.iloc[5171] #Just access the data using index location
index_value


# In[33]:


data=np.array([index_value]) #Convert 1D to 2D data


# In[34]:


pre_val=lr.predict(data) # predict the value
if pre_val[0]: #Set the check, got a result mail are spam or not
    print('Spam Mail')
else:
    print('Not Spam')


# In[35]:


df_email_drop


# # Implement three different types of algorithms, got a good score/accuracy of Logistic Regression Algorithm, So we can use Logistics Regression Algorithms
