import math
import numpy as np
from sklearn.metrics import confusion_matrix

def distance ( x , sample, single = False):
    
    if(single==True):
        sum = 0
        for j in range(len(sample)):
            sum += (x[j]-sample[j])**2
        return sum

    
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
#    print(labels)

    if(labels['Epilepsy']>=labels['Normal'] and labels['Epilepsy']>=labels['Nothing']):
        return 'Epilepsy'
    elif(labels['Normal']>=labels['Epilepsy'] and labels['Normal']>=labels['Nothing']):
        return 'Normal'
    elif(labels['Nothing']>=labels['Epilepsy'] and labels['Nothing']>=labels['Normal']):
        return 'Nothing'
    else:
        return 1



def kNearestNeighbors(X, y, X_test, y_test):
    
    print("K-Nearest Neighbors Algorithm")
    label1 = 0
    label3 = 0
    label5 = 0
    labels = [[],[],[]]
    
    for i in range(len(X_test)):
        d = distance(X, X_test[i])
        d = np.array(sorted(d,key=lambda q: q[1]))
        
        #        print("-----------(K=1)-----------")
        label = findLabel(d,y,1)
        labels[0].append(label)
        if(label == y_test[i]):
            label1 += 1;
        #        print("-----------(K=3)-----------")
        label = findLabel(d,y,3)
        labels[1].append(label)
        if(label == y_test[i]):
            label3 += 1
        #        print("-----------(K=5)-----------")
        label = findLabel(d,y,5)
        labels[2].append(label)
        if(label == y_test[i]):
            label5 += 1;


    print("Accuracy K=1",label1/len(y_test))
    print("Accuracy K=3",label3/len(y_test))
    print("Accuracy K=5",label5/len(y_test))
    print("--------")
    print("Confusion Matrix K=1")
    confusion = confusion_matrix( y_test , labels[0] )
    print(confusion)

    print("Confusion Matrix K=3")
    confusion = confusion_matrix( y_test , labels[1] )
    print(confusion)

    print("Confusion Matrix K=5")
    confusion = confusion_matrix( y_test , labels[2] )
    print(confusion)
    print("-------------------------")
