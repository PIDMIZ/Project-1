#!/usr/bin/env python
# coding: utf-8

#                                      Simple Linear Regression Project

# # Import Necessary Libraries

# In[1]:


import numpy as np #import numpy library
import pandas as pd #import pandas library
import seaborn as sns #import seaborn library
import matplotlib.pyplot as plt #import matplotlib library


# # Read data using pandas library

# In[2]:


df=pd.read_csv('C:\\Users\\muham\\Downloads\\Machine Learning Project\\Data Set of 1\\dataset1.txt') #read data using pandas function
df.shape #check the shape of data


# # df.head() Print first rows of data

# In[3]:


df.head() #print the first rows of data


# # #Data contain Categorical variable, First convert into One hot encoding 

# In[4]:


pd.get_dummies(df) #get the dummies values
#Acutally, dummies mean where categorical variables present convert to False and True
#In main dataframe, let sex contain two type of category (Male and female), get_dummies function make two columns,Similarly for every categorical data
#False mean value are 0 and True mean value are 1


# In[5]:


df_dummies=pd.get_dummies(df,drop_first=True) #Now remove every first level of dummies
df_dummies


# # Checking Data Clean or any null value present are not

# In[6]:


df_dummies.isnull().sum() #check any null values are present or not


# In[7]:


#Null values are not present


# # Visualize data using Scatter Plot

# In[8]:


sns.scatterplot(x='total_bill',y='tip', data=df) #Draw a scatter plot using seaborn library
plt.show()
#Graph give information, when total bill increase tip also increase


# In[9]:


sns.scatterplot(x='size',y='total_bill', data=df_dummies) #Draw a scatter plot using seaborn library
plt.show()
#Graph give information, when total bill increase t also increa


# # Split Data into X and Y

# In[10]:


x=df_dummies.drop('total_bill',axis=1) #Drop total_bill from data
y=df_dummies['total_bill'] #save drop values in y
print("Shape of x",x.shape) #check the shape of X data
print('shape of y',y.shape)#check the shape of y data


# # Split Data into Train test 

# In[11]:


from sklearn.model_selection import train_test_split #import train_test_split library from sklearn model


# In[12]:


X_train,X_test, y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=51) #split data into X_train,X_test,y_train,y_test 
print('Shape of X_train',X_train.shape) #check the shape of X_train
print('Shape of X_test',X_test.shape) #check the shape of X_test
print('Shape of y_train',y_train.shape) #check the shape of y_train
print('Shape of y_test',y_test.shape) #check the shape of y_test


# # Feature Scaling

# In[13]:


from sklearn.preprocessing import StandardScaler #import Standard Scaler fom sklearn .preprocessing model


# In[14]:


sc=StandardScaler() #Create object of standard scaler
sc.fit(X_train) #Fit Standard Scaling to fit data
X_train=sc.transform(X_train) #Transform X_train data
X_test=sc.transform(X_test) #Transfrom X_test Data


# # Linear Regression Model Training

# In[15]:


from sklearn.linear_model import LinearRegression #import Linear Regression model from sklearn 


# In[16]:


lr=LinearRegression() #Create object of Linear Regression model


# In[17]:


lr.fit(X_train,y_train)#Fit linear model to training data


# In[18]:


lr.coef_ #check the coefficient value of model i.e Beta 1 and Beta 2


# In[19]:


lr.intercept_ #check the value of a


# # Predicted value

# In[20]:


X_test[0,:] #Check the value of first bill value of text data


# In[21]:


lr.predict([X_test[0,:]]) #predict the value of First data


# In[22]:


lr.predict(X_test) #PRedict all the values


# In[23]:


y_test #compare value of X_test i.e predicted with y_test


# In[24]:


lr.score(X_test,y_test) #check the accuracy of model


# In[25]:


y_pred=lr.predict(X_test) #Take the predicted value od X_test and store to y_pred for model evaluation


# # Model Evaluation using R_Squared

# In[26]:


from sklearn.metrics import r2_score #import R_squared from sklearn metrics


# In[27]:


r2_score(y_test,y_pred) #Check the accuracy of model using R-squared

