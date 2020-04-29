def binary_quantization(x):
    for i in range(0,len(x)):
        for j in range(0, len(x[i])):
            if(x[i,j] > 127):
                x[i,j] = 1
            else:
                x[i,j] = 0



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

