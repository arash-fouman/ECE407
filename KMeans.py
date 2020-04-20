import pandas as pd
import numpy as np
import math
from random import random, seed

def read_file (filename):
    
    train = pd.read_csv(filename)
    x = train.iloc[:, 1:3].values
    y = train.iloc[: , 3].values
    return x, y

def dist( a , b ):

    if(type(a) is int):
        return abs(a-b)
    else:
        sum = 0
        for i in range(len(a)):
            sum += (a[i] - b[i])**2
        return math.sqrt(sum)

def cluster( x, mu ):

    classes = []
    for i in range(len(mu)):
        classes.append([])
    
    for i in x:
        d = []
        for j in range(len(mu)):
            d.append(dist(i,mu[j]))
        _min = d.index(min(d))
        classes[_min].append(i)
    return classes

def findMu ( c ):
    sum = [0,0]
    for i in c:
        sum[0] += i[0]
        sum[1] += i[1]
    sum[0] = sum[0]/len(c)
    sum[1] = sum[1]/len(c)
    return np.array(sum)

def Kmeans(x, mu):
    
    while(True):
        alaki = np.array([])
        mu_prime = [[] for i in mu]
        classes = cluster(x , mu)
        
        for i in range(len(mu_prime)):
            mu_prime[i] = np.array(findMu(classes[i]))
            alaki = np.append(alaki, (mu[i] == mu_prime[i]).all())

        print(alaki)
        if(alaki.all()):
            break
        else:
            mu = mu_prime
            
    return classes,mu_prime






#----------------------------- HW7 -------------------------------#

x  , y  = read_file("./HW6/Q1_Data_HW6.csv")
#seed(0)

mu = []
for i in range(4):
    mu.append(int(random()*len(x)))
print("indices of randomly selected centers:")
print(mu)
for i in range(len(mu)):
    mu[i] = x[mu[i]]
mu = np.array(mu)
print("randomly selected centers:")
print(mu)

print("------K-Means execution--------")
classes, mu = Kmeans(x, mu)
print("centers after K-Means: ")
print(np.array(mu))

d=[]
for i in mu:
    d.append(dist(i,[4.9,6.2]))
print("Distance of the samlpe from the centers: ",d)
print("Class of the samlpe: ",d.index(min(d)))


