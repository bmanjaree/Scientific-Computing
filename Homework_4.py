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


# In[3]:


#len(t)


# In[4]:


# (1,4)-accurate scheme (first-order accurate in time, fourth-order accurate in space)
G = lambda z: lambda_star*(32*np.cos(z)-2*np.cos(2*z)-30)*(1/12)+1


# In[5]:


A1 = copy.deepcopy(G(1))


# In[6]:


G_one = lambda z: np.abs(G(z)); # Define the function we want to maximize
index = scipy.optimize.fminbound(lambda z:-G_one(z), -np.pi,np.pi) # This gives the *maximizer*
maximum = G_one(index)
#print(maximum)


# In[7]:


A2 = copy.deepcopy(maximum)


# In[8]:


#D4
n = 128
#sparse matrix for derivative term
e1 = np.ones(n)
d_D4 = [-n+1,-n+2,-2,-1,0,1,2,n-2,n-1]
D4 = scipy.sparse.spdiags([16*e1,-e1,-e1,16*e1,-30*e1,16*e1,-e1,-e1,16*e1],d_D4,n,n)
#print(D4.todense())
D4 *= 1/12
#print(D4.todense())


# In[9]:


#D4.todense()


# In[10]:


#print(D4.toarray()[-1,0])
#plt.spy(D4)


# In[11]:


A3 = copy.deepcopy(D4.todense())
#A3


# In[12]:


#len(x)


# In[13]:


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


# In[14]:


#np.shape(D4)


# In[15]:


#usol_fe
A4 = 0


# In[16]:


#np.shape(usol_fe[:,-1].reshape(128,1))


# In[39]:


A5 = copy.deepcopy(usol_fe[:,-1].reshape(128,1))
#print(A5)


# In[40]:


# Crank Nicholoson scheme (first-order accurate in time, fourth-order accurate in space)
G_CN = lambda z: (1+lambda_star/2*(2*np.cos(z)-2))/(1-lambda_star/2*(2*np.cos(z)-2))


# In[41]:


G_one_CN = lambda z: np.abs(G_CN(z)); # Define the function we want to maximize
index_CN = scipy.optimize.fminbound(lambda z:-G_one_CN(z), -np.pi,np.pi) # This gives the *maximizer*
maximum_CN = G_one_CN(index_CN)
#print(maximum_CN)


# In[42]:


A6 = copy.deepcopy(maximum_CN)


# In[43]:


d_B = [-n+1,-1,0,1,n-1]
B = scipy.sparse.spdiags([(-lambda_star/2)*e1,(-lambda_star/2)*e1,(1+lambda_star)*e1,(-lambda_star/2)*e1,(-lambda_star/2)*e1],d_B,n,n)


# In[44]:


#B.todense()


# In[45]:


d_C = [-n+1,-1,0,1,n-1]
C = scipy.sparse.spdiags([(lambda_star/2)*e1,(lambda_star/2)*e1,(1-lambda_star)*e1,(lambda_star/2)*e1,(lambda_star/2)*e1],d_C,n,n)


# In[46]:


#C.todense()


# In[47]:


A7 = copy.deepcopy(B)


# In[48]:


A8 = copy.deepcopy(C)


# In[66]:


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


# In[67]:


A9 = copy.deepcopy(usol_CN[:,-1].reshape(128,1))
#A9


# In[68]:


#np.shape(A9)
#print(A9)


# In[69]:


#A9


# In[70]:


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


# In[71]:


A10 = copy.deepcopy(usol_CN_b[:,-1].reshape(128,1))


# In[77]:


exact_128 = np.genfromtxt('exact_128.csv', delimiter=',')
#np.shape(exact_128)
diff_128 = scipy.linalg.norm(A5-exact_128.reshape(128,1))
#diff_128


# In[78]:


#print(np.shape(A5))


# In[79]:


A11 = copy.deepcopy(diff_128)


# In[80]:


#exact_128_CN = np.genfromtxt('exact_128.csv', delimiter=',')
#np.shape(exact_128)
diff_128_CN = scipy.linalg.norm(A9-exact_128.reshape(128,1))
#diff_128_CN


# In[81]:


A12 = copy.deepcopy(diff_128_CN)


# In[ ]:





# In[ ]:





# In[ ]:




