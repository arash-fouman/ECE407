import pandas as pd
import numpy as np
import math


def read_file (filename):
    
    train = pd.read_csv(filename)
    x = train.iloc[:, 1:3].values
    y = train.iloc[: , 3].values
    return x, y

def distance ( x , sample):
    
    dist = []
    for i in range(len(x)):
        sum = 0
        for j in range(len(sample)):
            sum += (x[i][j]-sample[j])**2
        dist.append([i,math.sqrt(sum)])

    return np.array(dist)

def findLabel ( dist, y , k ):

    candids = dist[0:k]
    print(candids)
    labels = {0:0, 1:0}
    for i in candids:
        l = y[int(i[0])]
        labels[l] += 1
    print(labels)

    if(labels[0]>labels[1]):
        return 0
    else:
        return 1


#----------------------------- HW6 -------------------------------#

x  , y  = read_file("Q1_Data_HW6.csv")
sample = [4.9,6.2]
d = distance(x, sample)
d = np.array(sorted(d,key=lambda x: x[1]))
print("-----------(K=1)-----------")
print("Label: ",findLabel(d,y,1))
print("-----------(K=3)-----------")
print("Label: ",findLabel(d,y,3))
print("-----------(K=5)-----------")
print("Label: ",findLabel(d,y,5))

