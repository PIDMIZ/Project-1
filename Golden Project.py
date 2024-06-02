#!/usr/bin/env python
# coding: utf-8

#                                         #Car Price Prediction

# # Import libraries

# In[1]:


import pandas as pd #Import pandas library
from sklearn.ensemble import RandomForestRegressor #Import Random Forest Regressor


# In[2]:


df=pd.read_csv('C:\\Users\\muham\\Downloads\\Machine Learning Project\\Car\\CAR DETAILS FROM CAR DEKHO.csv') #Read data using pandas library


# In[3]:


df.head() #Print First rows of data


# In[4]:


df.shape #Check the shape of data


# In[5]:


df.info() #Check the details of data whcih type of data have


# In[6]:


df.isnull().sum() #Check any null value are present or not


# In[7]:


print(df.fuel.value_counts()) #Check which catgeory of data in fuel column
print(df.seller_type.value_counts()) #Check which catgeory of data in seller type column
print(df.transmission.value_counts()) #Check which catgeory of data in transmission column
print(df.owner.value_counts()) ##Check which catgeory of data in ownel column


# # Convert Categorical data into Numerical

# In[8]:


df.replace({'fuel':{'Diesel':0,'Petrol':1,'CNG':2,'LPG':3,'Electric':4}},inplace=True) #Replace Categroical to Numerical of Fuel column
df.replace({'seller_type':{'Individual':0,'Dealer':1,'Trustmark Dealer':2}},inplace=True) #Replace Categroical to Numerical of seller_type column
df.replace({'transmission':{'Manual':0,'Automatic':1}},inplace=True)#Replace Categroical to Numerical of transmission column
df.replace({'owner':{'First Owner':0,'Second Owner':1,'Third Owner':2,'Fourth & Above Owner':3,'Test Drive Car':4}},inplace=True)#Replace Categroical to Numerical of Owner column


# In[9]:


df.head() #print First rows of data


# # Convert Data into X and Y

# In[10]:


X=df.drop(['name','selling_price'],axis=1) #Drop Name and Selling price from data
Y=df['selling_price'] #Save selling price in Y


# # Split Data into Train test data

# In[11]:


from sklearn.model_selection import train_test_split #Import train test split from sklearn library


# In[12]:


X_train, X_test, Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=2) #SPlit data into X_train, X_test,Y_train,Y_test
print('Shape of X_train', X_train.shape) #Check shape of X_train
print('Shape of X_test',X_test.shape) #Check shape of X_test
print('Shape of Y_train',Y_train.shape) #Check shape of Y_train
print("Shape of Y_test", Y_test.shape) #Check Shape of Y_test


# # Implement Random Forest Regressor

# In[13]:


RF=RandomForestRegressor() #Create object of Model


# In[14]:


RF.fit(X_train,Y_train) #Train the data


# In[15]:


RF.score(X_test,Y_test) #Check the Score of Model


# # Predict the value

# In[16]:


X_test.iloc[1,:] #Check which kind of data on that location


# In[17]:


RF.predict([X_test.iloc[1,:]]) #Predict the value on that location


# In[18]:


Y_test.iloc[1] #Verify with Y_test


# In[19]:


RF.predict(X_test) #Predict all X_test data


# In[20]:


Y_test #Print all Y_test data


# In[21]:


y_pred=RF.predict(X_test) #Save Predicted value of all X_test to y_pred


# # Model Evaluation

# In[22]:


from sklearn.metrics import r2_score #Import r2score from sklearn


# In[23]:


r2_score(Y_test,y_pred) #Checking the score/Accuracy of model

