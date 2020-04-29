import math
import numpy as np

def distance ( x , sample):
    
    dist = []
    for i in range(len(x)):
        sum = 0
        for j in range(len(sample)):
            sum += (x[i][j]-sample[j])**2
        
        dist.append([i,math.sqrt(sum)])
    
    return np.array(dist)

def findLabel ( dist, y , k ):
    
    candidates = dist[0:k]
#    print(candids)
    labels = {'Epilepsy':0, 'Normal':0, 'Nothing':0 }
    for i in candidates:
        l = y[int(i[0])]
        labels[l] += 1
    print(labels)

    if(labels['Epilepsy']>=labels['Normal'] and labels['Epilepsy']>=labels['Nothing']):
        return 'Epilepsy'
    elif(labels['Normal']>=labels['Epilepsy'] and labels['Normal']>=labels['Nothing']):
        return 'Normal'
    elif(labels['Nothing']>=labels['Epilepsy'] and labels['Nothing']>=labels['Normal']):
        return 'Nothing'
    else:
        return 1

