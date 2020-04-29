import pandas as pd
import numpy as np
import math
import scipy.linalg as la
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

import LR
import KNN

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import NearestNeighbors
from scipy.special import softmax


def read_file (filename):
    
    train = pd.read_csv(filename)
    x = train.iloc[:,:-1].values
    y = train.iloc[: ,-1].values
    return x , y





#----------------------------- Project -------------------------------#

X, y = read_file("./Train.csv")
X_test, y_test = read_file("./Test.csv")

stdScalar = preprocessing.StandardScaler().fit(X)
X = stdScalar.transform(X)
X_test = stdScalar.transform(X_test)

#quantile_transformer = preprocessing.QuantileTransform().fit(X)
#X = quantile_transformer.transform(X)
#X_test = quantile_transformer.transform(X_test)

#X = preprocessing.power_transform(X,method = 'box-cox')
#X_test = preprocessing.power_transform(X_test,method = 'box-cox')


#X = preprocessing.normalize(X, norm='l2')
#X_test = preprocessing.normalize(X_test, norm='l2')


clf = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X,y)

#-------KNN-------#
y_label = y_test
label1 = 0
label3 = 0
label5 = 0
labels = [[],[],[]]

for i in range(len(X_test)):
    print(i)
#    print(y_label[i])

    alaki = np.array(clf.kneighbors([X_test[i]]))

    #---------Retrieve the labels of the nearest neighbors---------#
    alaki2 = [alaki[1,0].transpose(),alaki[0,0].transpose()]
    alaki2=np.array(alaki2).transpose()
    alaki2 = np.array(sorted(alaki2,key=lambda q: q[1])) #sort in ascending order

    label = KNN.findLabel(alaki2,y,5) #find the most common label among the nearest neighbors
    labels[2].append(label)
    if(y_label[i] == label):
        label5+=1


#--------Print the classification accuracy----------#
#print(label1/len(y_label))
#print(label3/len(y_label))
print(label5/len(y_label))

#print("Confusion Matrix K=1")
#confusion = confusion_matrix( y_label , labels[0] )
#print(confusion)
#
#print("Confusion Matrix K=3")
#confusion = confusion_matrix( y_label , labels[1] )
#print(confusion)

print("Confusion Matrix K=5")
confusion = confusion_matrix( y_label , labels[2] )
print(confusion)


#-------Logistic______#
#clf = LogisticRegression(solver = 'saga',multi_class = 'multinomial' )
#clf = MLPClassifier(solver = 'lbfgs',hidden_layer_sizes=(10,5) )
#clf = RandomForestClassifier(n_estimators = 100)
#clf = tree.DecisionTreeClassifier()

#clf.fit( X, y)
#print("classification accuracy: ",clf.score(X_test, y_test))
#
#w = clf.coef_
#x = X[0]
#probs = np.matmul(w,x)
#probs = np.transpose(probs)
#probs = softmax(probs)
#print(probs)
