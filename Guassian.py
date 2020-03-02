
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


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

def variance_multi_var( var_in , mu):
    
    res = 0;
    
    for i in range( len(var_in) ):
        temp = (var_in[i]-mu)
        temp = temp[:, np.newaxis]
        
        res += np.matmul( temp , temp.transpose() );

    return res/len(var_in);
                  

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.
        
        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.
        
        """
    
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    
    return np.exp(-fac / 2) / N


print("start")

print("### Question 1 ###")
lst_dice_df = {}
lst_dice_temp = list(map(float,input().split(',')))
lst_dice_sample = list(map(int,input().split(',')))

for i in range(len(lst_dice_sample)):
    lst_dice_df[lst_dice_sample[i]] = lst_dice_temp[i]

mu = expectation_discrete(lst_dice_sample, lst_dice_df)
print (mu)

E_x2 = expectation_discrete_var(lst_dice_sample, lst_dice_df)
variance = E_x2-mu**2;
print(E_x2)

check = input()
if( check != '#'):
    print("Error in Reading Q1.")

print("### Question 2 ###")

lst = list(map(int,input().split(',')))

if(len(lst) == 1):
    lst = list(map(int,input().split(',')))

mu = mean(lst)
variance = variance_from_data(lst, mu)
sigma = math.sqrt(variance)

print (lst);

print (mu)
print (variance)

check = input()
if( check != '#'):
    print("Error in Reading Q2.")

x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))


lst = []

while True:
    temp = list(input().split(' '))
    if ( temp[0] == '#'):
        break
    temp = [float(i) for i in temp]
    lst.append(temp)
lst = np.array(lst)
lst_tr = lst.transpose()

mu = np.zeros(len(lst_tr))
for i in range(len(mu)):
    mu[i] = mean(lst_tr[i])
covariance = variance_multi_var(lst,mu)

print(mu)
print(covariance)


# Our 2-dimensional distribution will be over variables X and Y
N = 60
X = np.linspace(0, 5, N)
Y = np.linspace(1, 6, N)
X, Y = np.meshgrid(X, Y)

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, covariance)

# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)

cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(27, -21)

plt.show()
