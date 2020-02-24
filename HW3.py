
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math



def mean( array_in ):
    
    sum = 0;
    for i in array_in:
        sum += i;
    sum /= len(array_in)
    return sum;

def variance_from_data ( array_in, mean ):
    
    res = 0;
    for i in array_in:
        res += (i-mean)**2;

    return res / len(array_in)

def expectation_discrete( samples, df ):
    
    res = 0;
    for x in samples:
        res += x*df[x]
    
    return res;

def expectation_discrete_var( samples, df ):
    
    res = 0;
    for x in samples:
        res += (x**2)*df[x]

    return res;

#def transpose( lst_in ):
#
#    lst_out = np.zeros(( len(lst_in[0]) , len(lst_in) ))
#
#    for i in range(len(lst_in)):
#        for j in range(len(lst_in[i])):
#            lst_out[j][i] = lst[i][j]
#    return lst_out
#

def variance_multi_var( var_in , mu):
    
    res = 0;
    
    for i in range( len(var_in) ):
        temp = (var_in[i]-mu)
        temp = temp[:, np.newaxis]
        
        res += np.matmul( temp , temp.transpose() );

    return res/len(var_in);
                  

#n = int(input("enter the number of inputs: "))
#lst = [];
#for i in range (n):
#    lst.append( int(input()) );

#lst = list(map(int,input("Enter sample data: ").split(',')))
#
#mu = mean(lst)
#variance = variance_from_data(lst, mu)
#sigma = math.sqrt(variance)
#
#print (lst);
#
#print (mu)
#print (variance)


#x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
#plt.plot(x, stats.norm.pdf(x, mu, sigma))
#plt.show()

print("start")

#lst_dice_df = {}
#lst_dice_temp = list(map(float,input("Enter prob. data: ").split(',')))
#lst_dice_sample = list(map(int,input("Enter sample data: ").split(',')))
#
#for i in range(len(lst_dice_sample)):
#    lst_dice_df[lst_dice_sample[i]] = lst_dice_temp[i]
#
#mu = expectation_discrete(lst_dice_sample, lst_dice_df)
#print (mu)
#
#E_x2 = expectation_discrete_var(lst_dice_sample, lst_dice_df)
#variance = E_x2-mu**2;
#print(E_x2)

n = int(input("Enter the number of vectors: "))
lst = []

print("Enter input vectors: ")
for i in range(int(n)):
    lst.append(list(map(float,input().split(' '))))
lst = np.array(lst)
print(lst)

print (lst.transpose())

lst_new = lst.transpose()

mu = np.zeros(len(lst_new))
for i in range(len(mu)):
    mu[i] = mean(lst_new[i])
print(mu)



covariance = variance_multi_var(lst,mu)

print(covariance)
