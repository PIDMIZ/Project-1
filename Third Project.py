#!/usr/bin/env python
# coding: utf-8

#                                   #Hyperparameter Tuning of ML Model

# # Import library

# In[1]:


import pandas as pd #import pandas library


# In[2]:


df=pd.read_csv('C:\\Users\\muham\\Downloads\\Machine Learning Project\\Hyper parameter\\emails.csv\\emails.csv') #Read the data using pandas library


# In[3]:


df.head()#print first rows of data


# In[4]:


df_drop_email_col=df.drop('Email No.',axis=1) #Drop Email No. Columns from data


# # Convert data into X and Y

# In[5]:


X=df_drop_email_col.drop('Prediction',axis=1)#drop prediction columns from data
Y=df_drop_email_col['Prediction'] #Prediction columns save into Y
print('Shape of X', X.shape) #print shape of X
print('Shape of Y', Y.shape) #Checkt the shape of Y


# # split data using train test split

# In[6]:


from sklearn.model_selection import train_test_split #from sklearn import train test split


# In[7]:


X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.2, random_state=52) #using train test split, split data into X_train, X_test, Y_train, Y_test
print('Shape of X train', X_train.shape)#Check the shape of X_train
print('Shape of X test', X_test.shape)#Check the shape of X_test
print('Shape of Y train', Y_train.shape)#Check the shape of Y_train
print('Shape of Y test', Y_test.shape)#Check the shape of Y_test



# # Just Implement SVM model

# In[8]:


from sklearn import svm #from sklearn import svm


# In[9]:


model=svm.SVC(kernel='rbf', C=30, gamma='auto') #implement SVC from smv with some hyperparamter
model.fit(X_train,Y_train) #Train the data
model.score(X_test,Y_test) #Check the score of data


# # Grid Search

# In[10]:


from sklearn.model_selection import GridSearchCV #Import GridSearchCV


# In[11]:


clf=GridSearchCV(svm.SVC(gamma='auto'),
                {   'C':[1,10,20],
                    'kernel':['rbf','linear']  },
                 cv=5,return_train_score=False) #implement GridSearchCV on SVM model with different Hyperparameter i.e C and kernels
clf.fit(X_train,Y_train) #Train the data
clf.cv_results_ #Checkt the result of model


# In[12]:


df_searchGrid=pd.DataFrame(clf.cv_results_) #Convert Result into Dataframe
df_searchGrid #print the data frame


# # Check the Score of Model

# In[13]:


df_searchGrid[['param_C','param_kernel','mean_test_score']] #print the value of C in one Columns, Second is kernel, third is mean test score


# In[14]:


clf.best_score_ #Check the socre of best model


# In[15]:


clf.best_estimator_ #Check the estimate which values are good fit


# In[16]:


clf.best_params_ #Check the best parameter that we use


# # Random Search

# In[17]:


from sklearn.model_selection import RandomizedSearchCV #import Randomized search CV from Sklearn


# In[18]:


rs=RandomizedSearchCV(svm.SVC(gamma='auto'),
                  {
                      'C':[1,10,20],
                      'kernel':['rbf','linear']
                  },
                  cv=5,
                   return_train_score=False,
                   n_iter=2
                  ) #implement GridSearchCV on SVM model with different Hyperparameter i.e C and kernels
rs.fit(X_train, Y_train) #Train the data
rs.cv_results_ #Check the result of model


# In[19]:


df_random=pd.DataFrame(rs.cv_results_) #Convert result into data frame


# In[20]:


df_random


# # Check the Score of Model

# In[21]:


df_random[['param_C','param_kernel','mean_test_score']]  #print the value of C in one Columns, Second is kernel, third is mean test score


# In[22]:


rs.best_score_ #Check the best score 


# In[23]:


rs.best_params_ #check the best parameter


# In[24]:


rs.best_estimator_ #Check the best estimator

