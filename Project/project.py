import pandas as pd
import numpy as np
import math
import scipy.linalg as la
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


import LR
import KNN
import Guassian

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import NearestNeighbors
from scipy.special import softmax

dataFrameLen = 25

def read_file (filename):
    
    train = pd.read_csv(filename)
    x = train.iloc[:,:-1].values
    y = train.iloc[: ,-1].values
    return x , y


def preprocess(x, single = False):
    
    if(single):
        lim = len(x)
        for i in range(lim-1):
            if(np.abs(x[i] - x[i+1]) > 1 or  lim-i < dataFrameLen):
                print(i)
                return x[i:idataFrameLen]

    arr = []
    lim = len(x[0])
    for k in x:
        for i in range(lim-1):
            if(np.abs(k[i] - k[i+1]) > 1 or  lim-i < 131):
                arr.append( k[i:i+dataFrameLen] )
#                print(x[i-1:i+100])
                break


    return np.array(arr)


#----------------------------- Project -------------------------------#

#X, y = read_file("./Train_2.csv")
#X_test, y_test = read_file("./Test_2.csv")
#
#dataFrameLen = 150
#X1=preprocess(X)
##X_test=preprocess(X_test)
#
#while True:
#    print("enter index: ")
#    index = int(input())
#    if(index == -1):
#        break
#    plt.plot(X[index])
#    plt.plot(X1[index],'r')
#    plt.ylim([-1,5])
#    plt.title(y[index])
#    plt.show()

##-------Guassian-------#
#dataFrameLen = 25
#X, y = read_file("./Train_2.csv")
#X_test, y_test = read_file("./Test_2.csv")
#
#X=preprocess(X)
#X_test=preprocess(X_test)
#
#
#Guassian.guassian( X, y, X_test, y_test)
#
##-------KNN-------#
#dataFrameLen = 24
#X, y = read_file("./Train_2.csv")
#X_test, y_test = read_file("./Test_2.csv")
#
#X=preprocess(X)
#X_test=preprocess(X_test)
#
#KNN.kNearestNeighbors(X,y,X_test,y_test)
#
#

##-------Logistic______#
dataFrameLen = 16
X, y = read_file("./Train_2.csv")
X_test, y_test = read_file("./Test_2.csv")

X=preprocess(X)
X_test=preprocess(X_test)

LR.logisticRegression( X, y, X_test, y_test)





#clf = LogisticRegression(solver = 'saga',multi_class = 'multinomial' )
#clf = MLPClassifier(solver = 'lbfgs',hidden_layer_sizes=(20,15,10) )
##clf = RandomForestClassifier(n_estimators = 100)
#clf = tree.DecisionTreeClassifier()
#
#clf.fit( X, y)
#alaki = clf.predict(X_test)
#print("classification accuracy: ",clf.score(X_test, y_test))
#confusion = confusion_matrix( y_test , alaki )
#print(confusion)

##
##w = clf.coef_
##x = X[0]
##probs = np.matmul(w,x)
##probs = np.transpose(probs)
##probs = softmax(probs)
##print(probs)
