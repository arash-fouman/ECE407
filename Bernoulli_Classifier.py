# MNIST alaki

import pandas as pd
import numpy as np
import collections
import math
import matplotlib.pyplot as plt

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


def priorProb(y):
    labels = np.zeros(10);
    for i in y:
        labels[i] = labels[i]+1
    return labels/len(y)


def pixel_probability( x, y ):
    
    a = collections.Counter(y)
    prob = np.zeros((10,784))

    for i in range (0,len(y)):
        prob[y[i],:] = prob[y[i],:] + x[i,:]
    
    for i in range(len(prob)):
        for j in range(len(prob[i])):
            if(prob[i,j] == 0):
                prob[i,j] = 1.0/(a[i]+10)
            else:
                prob[i,j] = prob[i,j]/a[i]

    
    return prob

def classify( x, pi_j, p_j  ):

    sum = math.log2(pi_j);
    for i in range(len(x)):
        sum += x[i]*math.log2(p_j[i])+(1-x[i])*math.log2(1-p_j[i])
    return sum

def max ( x ):
    
    maxClass = -65536
    index = 0
    for i in range(len(x)):
        if( x[i] > maxClass ):
            index = i
            maxClass = x[i]
    return index

def test( x , y , pi, pixel_prob ):
    
    post = []
    miss = []
    correct = 0;
    list = []
    for k in range(len(x)):
        print(k)
        arg =[]
        for j in range (0,10):
            arg.append(classify(x[k], pi[j], pixel_prob[j]))
        _class = max(arg)
        if( _class == y[k]):
            correct += 1
        else:
            print("***************")
            print(k)
            print(y[k])
            print(arg)
                    
        print( _class, " -- ", y[k])

    print(correct/len(y))

def display(x , y, i):
    
    img = x[i,:]
    plt.title("label %d"% (y[i]))
    plt.imshow(img.reshape((28,28)), cmap = plt.cm.gray_r)


#x_train , y_train = read_file("mnist_train.csv")
x_test  , y_test  = read_file("mnist_test.csv")
#
#binary_quantization(x_train)
binary_quantization(x_test)

#pi = priorProb(y_train)

#print(pi)

#pixel_prob = pixel_probability(x_train, y_train)

#test( x_test, y_test, pi, pixel_prob)


#print(pixel_prob[0])

display(x_test , y_test, 3893)

plt.show()

