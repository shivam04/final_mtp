
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import glob, os, sys
import time
import pickle
import datetime


# In[2]:


df = pd.read_csv('finalset/patient_data_48_hours.csv')
df.head(50)


# In[3]:


matrix3D = np.array(df.drop(['SUBJECT_ID', 'TimeStamp'], 1))
print(matrix3D)


# In[4]:


print(matrix3D.shape)


# In[5]:


matrix3D = np.array(matrix3D).reshape((6587, 49, 37))
print(matrix3D.shape)


# In[6]:


outcomes = pd.read_csv('finalset/outcomes.csv')
print(outcomes.head(10))


# In[7]:


Y = np.array(outcomes.drop(['SUBJECT_ID'], 1))
print(Y.shape)


# In[8]:


print("X,Y")
X = matrix3D
print(X.shape)
X = np.concatenate((X,X))
print(X.shape)
Y = np.concatenate((Y,Y))
print(Y.shape)
# In[9]:


X_train = X[:10540]
X_test = X[10540:]
Y_train = Y[:10540]
Y_test = Y[10540:]
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[10]:





# In[11]:

def model_op(X_tr,X_te):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_tr.shape[1],X_tr.shape[2]),activation='tanh'))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(25,activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adamax', metrics=['accuracy'])
    model.fit(X_tr, Y_train, validation_data=(X_te, Y_test), epochs=30,batch_size=8,verbose=1)
    scores = model.evaluate(X_te, Y_test, verbose=0)
    return scores[1]

class LSTM(object):
    def __init__(self):
        # define lower bound of benchmark function
        self.Lower = 0
        # define upper bound of benchmark function
        self.Upper = 1

    # function which returns evaluate function
    def function(self):
        def evalute(D,sol):
            sol = np.array(sol)
            op = sol>=0.5
            X_tr = X_train[:,:,op]
            X_te = X_test[:,:,op]
            cost = -1*model_op(X_tr,X_te)
            return cost
        return evalute


# In[13]:

from NiaPy.algorithms.basic import GreyWolfOptimizer
ans = []
for i in range(10):
    algorithm = GreyWolfOptimizer(D=37, NP=370, nFES=10, benchmark=LSTM())
    best = algorithm.run()
    print(best)
    ans.append(-1*best[1])
print(ans)
print(max(ans))

