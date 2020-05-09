import numpy as np
import math
import scipy.linalg as la
from sklearn.metrics import confusion_matrix


def mean( array_in ):
    
    res = np.zeros(len(array_in[0]))
    for j in array_in:
        res += j
    return res/len(array_in);

def expectation_discrete( samples, df ):
    
    res = 0;
    for x in samples:
        res += (x*df)
    return res;


def cov (x_in, mu ):
    
    x_temp = []
    for i in x_in:
        alaki = np.array([i-mu])
        x_temp.append(np.matmul(alaki.transpose(), alaki))
    
    Ex = expectation_discrete(x_temp, 1/len(x_temp))
    return Ex


def scale(X):
    
    a = []
    for i in X:
        a.append(i/3.866080156)
    return np.array(a)

def likelihood ( x, mean, cov ):
    
    determinant = abs(np.linalg.det(cov))
    inverse     = np.linalg.inv(cov)
    
    const_pi = (2*math.pi)**len(mean)
    denominator = math.sqrt( const_pi * determinant )
    a = 1/(denominator)
    
    b = np.array([x - mean])
    c = np.matmul(b,inverse)
    d = -np.matmul(c,x-mean)/2
    
    return math.log(a) + d

def label ( x ):
    if( x[0] > x[1] and x[0] > x[2] ):
        return 'Epilepsy'
    if( x[1] > x[0] and x[1] > x[2] ):
        return 'Normal'
    if( x[2] > x[1] and x[2] > x[0] ):
        return 'Nothing'
    return 'ridi'


def guassian( X, y, X_test, y_test):

    print("Guassian Algorithm")

    X = scale(X)
    X_test = scale(X_test)
    
    print(X.shape)
    mu = [[],[],[]]
    mu[0] = mean(X[0:80])
    mu[1] = mean(X[80:160])
    mu[2] = mean(X[160:240])
    
    #-------for test------#
#    print(mu)
    #---------------------#
    tah = 11
    sar = 1
    COV = [[],[],[]]
    alaki = np.array(X[0:80]).transpose()[sar:tah]
    alaki = alaki.transpose()
    COV[0] = cov(alaki,mu[0][sar:tah])
    
    alaki = np.array(X[80:160]).transpose()[16:26]
    alaki = alaki.transpose()
    COV[1] = cov(alaki,mu[0][16:26])
    
    COV[2] = cov(X[160:240],mu[2])
    
    ##-------for test------#
    #res = [0,0,0]
    #res[0] = (likelihood(X_test[25][sar:tah], mu[0][sar:tah], COV[0]))
    #res[1] = (likelihood(X_test[25][16:26], mu[1][16:26], COV[1]))
    #res[2] = (likelihood(X_test[25], mu[2], COV[2]))
    #
    #print(likelihood(X_test[0][sar:tah], mu[0][sar:tah], COV[0]))
    #print(likelihood(X_test[0][16:26], mu[1][16:26], COV[1]))
    #print(likelihood(X_test[0], mu[2], COV[2]))
    #print(label(res))
    ##---------------------#
    
    acc = 0
    alaki = []
    for index in range(len(X_test)):
        
        res = [ [] , [], []]
        res[0] = likelihood(X_test[index][sar:tah], mu[0][sar:tah], COV[0])
        res[1] = likelihood(X_test[index][16:26], mu[1][16:26], COV[1])
        res[2] = likelihood(X_test[index], mu[2], COV[2])
#        print(index,y_test[index],label(res))
        alaki.append(label(res))
        if(y_test[index] == label(res)):
            acc += 1

    print(acc/len(X_test))
    confusion = confusion_matrix( y_test , alaki )
    print(confusion)

