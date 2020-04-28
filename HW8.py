import pandas as pd
import numpy as np
import math
import scipy.linalg as la

def read_file (filename):
    
    train = pd.read_csv(filename)
    x = train.iloc[: ,0:2].values
#    y = train.iloc[: ,2].values
    return x

def mean(x):
    
    sum = 0;
    for i in x:
        sum += i
    return sum/len(x)



def cov(x):
    mu = [0, 0]
    mu[0] = mean(x[:,0])
    mu[1] = mean(x[:,1])
    
    temp = []
    for i in x:
        temp.append(i-mu)
    
    temp_t = np.array(temp).transpose()

    covarience = np.matmul(temp_t,temp)/len(x)
    return covarience



#----------------------------- HW8 -------------------------------#

x = read_file("./data2.csv")

covariance = cov(x) #the covariance matrix
eigenValues , eigenVectors = la.eig(covariance) #the eigen values and eigen vectors matrices

print("Covariance:")
print(covariance)
print("------------\nEigen Values:")
print(eigenValues)
print("------------\nEigen Vectors:")
print(eigenVectors)




