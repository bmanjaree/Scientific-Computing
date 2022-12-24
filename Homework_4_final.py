#!/usr/bin/env python
# coding: utf-8

# ## AMATH Scientific Computing
# ### Homework-4
# #### Manjaree Binjolkar

# In[1]:


import numpy as np
import scipy.sparse 
import matplotlib.pyplot as plt
import pdb
import numpy.matlib
import time
import copy

from matplotlib import animation, rc
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import splu
from scipy.sparse import csr_matrix, triu, spdiags

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv


# 1. 1. We will start by using 129 equally spaced points between [􀀀L; L] (including L),
# and then removing L so that your computational domain has 128 equally spaced
# points. This de nes  x. Use 501 equally spaced points between [0; 2] (including
# the endpoints). This defines  t. Calculate the CFL number and call it   .

# In[2]:


m = 129
L = 10
x1 = np.linspace(-L, L, m)
x = x1[:-1]
delta_x = x[1] - x[0]
#print(delta_x)
#print(np.shape(delta_x))
t = np.linspace(0, 2, 501, endpoint = True)
delta_t = t[1] - t[0]
#print(delta_t)
#print(np.shape(delta_t))
#print(t)
#print(np.shape(t))
#calculate the CFL number λ and call it λ∗
alpha = 2
lambda_star = alpha*(delta_t)/(delta_x**2)
#print(lambda_star)
#print(np.shape(lambda_star))

# (1,4)-accurate scheme (first-order accurate in time, fourth-order accurate in space)
G = lambda z: lambda_star*(32*np.cos(z)-2*np.cos(2*z)-30)*(1/12)+1

A1 = copy.deepcopy(G(1))

G_one = lambda z: np.abs(G(z)); # Define the function we want to maximize
index = scipy.optimize.fminbound(lambda z:-G_one(z), -np.pi,np.pi) # This gives the *maximizer*
maximum = G_one(index)
#print(maximum)

A2 = copy.deepcopy(maximum)

#D4
n = 128
#sparse matrix for derivative term
e1 = np.ones(n)
d_D4 = [-n+1,-n+2,-2,-1,0,1,2,n-2,n-1]
D4 = scipy.sparse.spdiags([16*e1,-e1,-e1,16*e1,-30*e1,16*e1,-e1,-e1,16*e1],d_D4,n,n)
#print(D4.todense())
D4 *= 1/12
#print(D4.todense())

A3 = copy.deepcopy(D4.todense())

#Forward Euler
#initial conditions - FE
Time = 2
dt = delta_t
time_steps = int(Time/dt)
usol_fe = np.zeros((len(x),len(t))) #placeholder for solution - more effcient
#u0 = np.exp(-x**2).T # Gaussian
u0 = 10*np.cos((2*np.pi*x)/L) + 30*np.cos((8*np.pi*x)/L)
usol_fe[:,0] = u0
u1 = u0
CFL = lambda_star
for j in range(time_steps):
    u2 = u1 + CFL*(D4@u1)
    u1 = u2
    usol_fe[:,j+1] = u2
    
A4 = 0

A5 = copy.deepcopy(usol_fe[:,-1].reshape(128,1))


# In[3]:


# Crank Nicholoson scheme (first-order accurate in time, fourth-order accurate in space)
G_CN = lambda z: (1+lambda_star/2*(2*np.cos(z)-2))/(1-lambda_star/2*(2*np.cos(z)-2))

G_one_CN = lambda z: np.abs(G_CN(z)); # Define the function we want to maximize
index_CN = scipy.optimize.fminbound(lambda z:-G_one_CN(z), -np.pi,np.pi) # This gives the *maximizer*
maximum_CN = G_one_CN(index_CN)
#print(maximum_CN)

A6 = copy.deepcopy(maximum_CN)

d_B = [-n+1,-1,0,1,n-1]
B = scipy.sparse.spdiags([(-lambda_star/2)*e1,(-lambda_star/2)*e1,(1+lambda_star)*e1,(-lambda_star/2)*e1,(-lambda_star/2)*e1],d_B,n,n)

d_C = [-n+1,-1,0,1,n-1]
C = scipy.sparse.spdiags([(lambda_star/2)*e1,(lambda_star/2)*e1,(1-lambda_star)*e1,(lambda_star/2)*e1,(lambda_star/2)*e1],d_C,n,n)

A7 = copy.deepcopy(B.todense())

A8 = copy.deepcopy(C.todense())


#initial conditions - Crank Nicholson
Time = 2
dt = delta_t
time_steps = int(Time/dt)
usol_CN = np.zeros((len(x),len(t))) #placeholder for solution - more effcient
#u0 = np.exp(-x**2).T # Gaussian
u0 = 10*np.cos((2*np.pi*x)/L) + 30*np.cos((8*np.pi*x)/L)
usol_CN[:,0] = u0
u1 = u0
#CFL = lambda_star
solver = scipy.sparse.linalg.splu(B)
for j in range(time_steps):
    
    u2 = solver.solve(C@u1)
    
    u1 = u2
    usol_CN[:,j+1] = u2

A9 = copy.deepcopy(usol_CN[:,-1].reshape(128,1))
#A9

##bigscstab
#initial conditions - Crank Nicholson
Time = 2
dt = delta_t
time_steps = int(Time/dt)
usol_CN_b = np.zeros((len(x),len(t))) #placeholder for solution - more effcient
#u0 = np.exp(-x**2).T # Gaussian
u0 = 10*np.cos((2*np.pi*x)/L) + 30*np.cos((8*np.pi*x)/L)
usol_CN_b[:,0] = u0
u1 = u0
CFL = lambda_star
for j in range(time_steps):
    
    x, exitcode = scipy.sparse.linalg.bicgstab(B, C@u1)
    #solver = scipy.sparse.linalg.splu(B)
    #u2 = solver.solve(C@u1)
    u2 = x
    u1 = u2
    usol_CN_b[:,j+1] = u2
    
A10 = copy.deepcopy(usol_CN_b[:,-1].reshape(128,1))

exact_128 = np.genfromtxt('exact_128.csv', delimiter=',')
#np.shape(exact_128)
diff_128 = scipy.linalg.norm(A5-exact_128.reshape(128,1))
#diff_128

A11 = copy.deepcopy(diff_128)

#exact_128_CN = np.genfromtxt('exact_128.csv', delimiter=',')
#np.shape(exact_128)
diff_128_CN = scipy.linalg.norm(A9-exact_128.reshape(128,1))
#diff_128_CN

A12 = copy.deepcopy(diff_128_CN)


# In[4]:


m = 257
L = 10
x1 = np.linspace(-L, L, m)
x = x1[:-1]
delta_x = x[1] - x[0]
#print(delta_x)
#print(np.shape(delta_x))
t = np.linspace(0, 2, (500*4)+1, endpoint = True)
delta_t = t[1] - t[0]
#print(delta_t)
#print(delta_t)
#print(np.shape(delta_t))
#print(t)
#print(np.shape(t))
#calculate the CFL number λ and call it λ∗
alpha = 2
lambda_star = alpha*(delta_t)/(delta_x**2)
#print(lambda_star)
#print(np.shape(lambda_star))

# (1,4)-accurate scheme (first-order accurate in time, fourth-order accurate in space)
G = lambda z: lambda_star*(32*np.cos(z)-2*np.cos(2*z)-30)*(1/12)+1

#A1 = copy.deepcopy(G(1))

G_one = lambda z: np.abs(G(z)); # Define the function we want to maximize
index = scipy.optimize.fminbound(lambda z:-G_one(z), -np.pi,np.pi) # This gives the *maximizer*
maximum = G_one(index)
#print(maximum)

#A2 = copy.deepcopy(maximum)

#D4
n = 256
#sparse matrix for derivative term
e1 = np.ones(n)
d_D4 = [-n+1,-n+2,-2,-1,0,1,2,n-2,n-1]
D4 = scipy.sparse.spdiags([16*e1,-e1,-e1,16*e1,-30*e1,16*e1,-e1,-e1,16*e1],d_D4,n,n)
#print(D4.todense())
D4 *= 1/12
#print(D4.todense())

#A3 = copy.deepcopy(D4.todense())

#Forward Euler
#initial conditions - FE
Time = 2
dt = delta_t
time_steps = int(Time/dt)
usol_fe = np.zeros((len(x),len(t))) #placeholder for solution - more effcient
#u0 = np.exp(-x**2).T # Gaussian
u0 = 10*np.cos((2*np.pi*x)/L) + 30*np.cos((8*np.pi*x)/L)
usol_fe[:,0] = u0
u1 = u0
CFL = lambda_star
for j in range(time_steps):
    u2 = u1 + CFL*(D4@u1)
    u1 = u2
    usol_fe[:,j+1] = u2
    
#A4 = 0

A5_256 = copy.deepcopy(usol_fe[:,-1].reshape(256,1))


# In[5]:


#print(A5_256)


# In[6]:


# Crank Nicholoson scheme (first-order accurate in time, fourth-order accurate in space)
G_CN = lambda z: (1+lambda_star/2*(2*np.cos(z)-2))/(1-lambda_star/2*(2*np.cos(z)-2))

G_one_CN = lambda z: np.abs(G_CN(z)); # Define the function we want to maximize
index_CN = scipy.optimize.fminbound(lambda z:-G_one_CN(z), -np.pi,np.pi) # This gives the *maximizer*
maximum_CN = G_one_CN(index_CN)
#print(maximum_CN)

#A6 = copy.deepcopy(maximum_CN)

d_B = [-n+1,-1,0,1,n-1]
B = scipy.sparse.spdiags([(-lambda_star/2)*e1,(-lambda_star/2)*e1,(1+lambda_star)*e1,(-lambda_star/2)*e1,(-lambda_star/2)*e1],d_B,n,n)

d_C = [-n+1,-1,0,1,n-1]
C = scipy.sparse.spdiags([(lambda_star/2)*e1,(lambda_star/2)*e1,(1-lambda_star)*e1,(lambda_star/2)*e1,(lambda_star/2)*e1],d_C,n,n)

#A7 = copy.deepcopy(B.todense())

#A8 = copy.deepcopy(C.todense())


#initial conditions - Crank Nicholson
Time = 2
dt = delta_t
time_steps = int(Time/dt)
usol_CN = np.zeros((len(x),len(t))) #placeholder for solution - more effcient
#u0 = np.exp(-x**2).T # Gaussian
u0 = 10*np.cos((2*np.pi*x)/L) + 30*np.cos((8*np.pi*x)/L)
usol_CN[:,0] = u0
u1 = u0
#CFL = lambda_star
solver = scipy.sparse.linalg.splu(B)
for j in range(time_steps):
    
    u2 = solver.solve(C@u1)
    
    u1 = u2
    usol_CN[:,j+1] = u2

A9_256 = copy.deepcopy(usol_CN[:,-1].reshape(256,1))
#A9

exact_256 = np.genfromtxt('exact_256.csv', delimiter=',')
#np.shape(exact_128)
diff_256 = scipy.linalg.norm(A5_256-exact_256.reshape(256,1))
#diff_128

A13 = copy.deepcopy(diff_256)

#exact_128_CN = np.genfromtxt('exact_128.csv', delimiter=',')
#np.shape(exact_128)
diff_256_CN = scipy.linalg.norm(A9_256-exact_256.reshape(256,1))
#diff_128_CN

A14 = copy.deepcopy(diff_256_CN)


# In[ ]:




