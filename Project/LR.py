from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.special import softmax
from sklearn.metrics import confusion_matrix

def getMaxProb( p ):
    
    index_vec = []
    for i in range(len(p)):
        m = max(p[i])
        index = np.where(p[i] == m)
        index_vec.append(index[0])
    return index_vec


def findLabel ( labels ):
    
    l = []
    for i in labels:
        if(i == 0):
            l.append('Epilepsy')
        elif(i == 1):
            l.append('Normal')
        elif(i == 2):
            l.append('Nothing')
    return l


def classify ( w , x ):
    
    probs = np.matmul(w,x)
    probs = np.transpose(probs)
    probs = softmax(probs)
    
    return findLabel(getMaxProb(probs))

def logisticRegression( X, y, X_test, y_test):
    
    print("Logistic Regression Algorithm")
    clf = LogisticRegression(solver = 'saga',multi_class = 'multinomial', max_iter = 60 )
    clf.fit( X, y )

    w = clf.coef_
    labels = classify ( w, X_test.transpose() )
    acc = 0
    for i in range(len(labels)):
        if(y_test[i] == labels[i]):
            acc += 1
    print(acc/len(y_test))
    confusion = confusion_matrix( y_test , labels )
    print(confusion)

