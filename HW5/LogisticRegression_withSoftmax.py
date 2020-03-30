import sys
import pandas as pd
import numpy as np
import collections
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax
from sklearn.metrics import confusion_matrix

np.set_printoptions(threshold=sys.maxsize)

def read_file (filename):
    
    train = pd.read_csv(filename)
    x = train.iloc[:, 1:].values
    y = train.iloc[: , 0].values
    return x, y

def binary_quantization(x):
    for i in range(0,len(x)):
        for j in range(0, len(x[i])):
            if(x[i,j] > 127):
                x[i,j] = 1
            else:
                x[i,j] = 0


def display(x , y, i):
    
    img = x[i,:]
    plt.title("label %d"% (y[i]))
    plt.imshow(img.reshape((28,28)), cmap = plt.cm.gray_r)


def getMaxProb( p ):

    index_vec = []
    for i in range(len(p)):
        max = -1
        for j in range(len(p[i])):
            if( p[i][j] > max ):
                index = j
                max = p[i][j]
        index_vec.append(index)
    return index_vec



def classify ( w , x ):
    
    probs = np.matmul(w,x)
    probs = np.transpose(probs)
    probs = softmax(probs)

    return getMaxProb(probs)


#------------ Start of the Code--------------#
x_train , y_train = read_file("mnist_train.csv")
binary_quantization(x_train)
x_test  , y_test  = read_file("mnist_test.csv")
binary_quantization(x_test)

clf = LogisticRegression(solver = 'saga',multi_class = 'multinomial' )
clf.fit( x_train, y_train)

print("classification accuracy: ",clf.score(x_test, y_test))

w = np.array(clf.coef_)
np.save("alaki",w)
#w = np.load("alaki.npy")
print("W vector shape: ",w.shape)

x = np.transpose(x_test)
print("test vector shape: ",x.shape)
#
classes = classify(w , x)

print("Confusion Matrix")
confusion = confusion_matrix( y_test , classes )
print(confusion)


#display(x_test , y_test, 3893)

#plt.show()
