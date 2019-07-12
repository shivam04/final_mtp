import pandas as pd
import numpy as np
import glob, os, sys
import time
import pickle
import datetime
df = pd.read_csv('finalset/patient_data_24_hours.csv')
print(df.head(50))
matrix3D = np.array(df.drop(['SUBJECT_ID', 'TimeStamp'], 1))
print(matrix3D)
print(matrix3D.shape)
matrix3D = np.array(matrix3D).reshape((6587, 25, 37))
print(matrix3D.shape)
outcomes = pd.read_csv('finalset/outcomes.csv')
print(outcomes.head(10))
Y = np.array(outcomes.drop(['SUBJECT_ID'], 1))
print(Y.shape)
X = matrix3D
print(X.shape)
X = np.concatenate((X,X))
print(X.shape)
Y = np.concatenate((Y,Y))
print(Y.shape)
X_train = X[:10540]
X_test = X[10540:]
Y_train = Y[:10540]
Y_test = Y[10540:]
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences=True))
model.add(LSTM(25))
model.add(Dropout(0.2))
model.add(Dense(100,activation='tanh'))
model.add(Dense(25,activation='tanh'))
model.add(Dense(10,activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print(model.summary())
#model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30,batch_size=8,verbose=1)
#scores = model.evaluate(X_test, Y_test)
#print(scores[1]*100)
def sigmoid(x):
    return 1/(1+np.exp(-x))
ans = []
class LSTM(object):
    def __init__(self):
        # define lower bound of benchmark function
        self.Lower = 0
        # define upper bound of benchmark function
        self.Upper = 1

    # function which returns evaluate function
    def function(self):
        def evalute(D,sol):
            from keras.models import Sequential
            from keras.layers import Dense
            from keras.layers import LSTM
            from keras.layers import Dropout
            sol = np.array(sol)
            sol = sigmoid(sol)
            #display(sol)
            op = sol>=0.5
            X_tr = X_train[:,:,op]
            X_te = X_test[:,:,op]
            modelf = Sequential()
            modelf.add(LSTM(100, input_shape=(X_tr.shape[1],X_tr.shape[2]),return_sequences=True))
            modelf.add(LSTM(50))
            modelf.add(Dropout(0.2))
            modelf.add(Dense(100,activation='tanh'))
            modelf.add(Dense(25,activation='tanh'))
            modelf.add(Dense(10,activation='tanh'))
            modelf.add(Dense(1, activation='sigmoid'))
            modelf.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
            modelf.fit(X_tr, Y_train, validation_data=(X_te, Y_test), epochs=30,batch_size=50,verbose=2)
            scores = modelf.evaluate(X_te, Y_test, verbose=0)
            k = -1*scores[1]
            return k
        return evalute
from NiaPy.algorithms.basic import GreyWolfOptimizer
for i in range(10):
    algorithm = GreyWolfOptimizer(D=37, NP=370, nFES=10, benchmark=LSTM())
    best = algorithm.run()
    print(-1*best[1])
    ans.append(-1*best[1])
print(ans)
print(max(ans))
