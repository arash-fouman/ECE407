#Perceptron Training Algorithm

import numpy as np

def inserBias( x ):
    for i in range(len(x)):
        x[i].insert(0,1)
    return np.array(x)

def predict (w,x):
    return np.sign(np.matmul(x,w))

x = ([[1,3],[2,3],[-1,1],[-2,0.5]])
y = ([1, 1, -1, -1])
w = np.array([0, 0, 0])

x = inserBias(x)

epoch = 0
while epoch< 10:
    w_temp = w
    print("Epoch: ",epoch)
    for i in range(len(x)):
        y_t = np.sign(np.dot(x[i],w))
        if( y_t != y[i]):
            w = w + y[i]*x[i]
    if((w_temp == w)).all():
        break
    epoch += 1

print("***finished training***")
print("weight vector: ",w)

print("class of the given input: ", predict(w, inserBias([[1,-1]]) ) )

