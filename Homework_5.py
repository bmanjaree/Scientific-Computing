#!/usr/bin/env python
# coding: utf-8

# ## AMATH Scientific Computing
# ### Homework-5
# #### Manjaree Binjolkar

# In[1]:


import numpy as np
import numpy.matlib
import scipy.integrate
import scipy.sparse
import scipy.sparse.linalg
import mpl_toolkits.mplot3d
from scipy.fft import fft, ifft, fft2, ifft2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy


# In[2]:



def cheb(N):
    # N is the number of points in the interior.
    if N==0:
        D = 0
        x = 1
        return D, x
    vals = np.linspace(0, N, N+1)
    x = np.cos(np.pi*vals/N)
    x = x.reshape(-1, 1)
    c = np.ones(N-1)
    c = np.pad(c, (1,), constant_values = 2)
    c *= (-1)**vals
    c = c.reshape(-1, 1)
    X = np.tile(x, (1, N+1))
    dX = X-X.T                  
    D  = (c*(1/c.T))/(dX+(np.eye(N+1)))       #off-diagonal entries
    D  = D - np.diag(sum(D.T))                #diagonal entries

    return D, x


# ## Problem 1
# In both of our solves we will use x,y ∈[−10,10], β = 1, D1 = D2 = 0.1, and will solve
# for t ∈ [0,25] with a step size of ∆t = 0.5.

# In[3]:


tspan = np.arange(0,25.5,0.5)

#Define the drag term
nu = 0.001
L=20
n=64

beta = 1
D1 = 0.1
D2 = 0.1

#Setup our x and y domain
x2 = np.linspace(-L/2,L/2,n+1) 
x=x2[0:-1] 
y=x.copy()

#Setup the x values.
r1=np.arange(0,n/2,1)
r2=np.arange(-n/2,0,1)
kx = (2*np.pi/L)*np.concatenate((r1,r2))
ky=kx.copy()

#Put X and Y on a meshgrid
[X,Y]=np.meshgrid(x,y)

#Do the same for the k values
[KX,KY]=np.meshgrid(kx,ky)

#This term shows up in the Laplacian 
#-->don't need this since I am directly using KX and KY
#K = KX**2+KY**2
#Kvec = K.reshape(n**2)#take transpose in function


# In[4]:


X, Y = np.meshgrid(x,y)
m = 3
alpha = 0

#initial conditions
u = (np.tanh(np.sqrt(X**2+Y**2))-alpha)*np.cos(m*np.angle(X+1j*Y) - np.sqrt(X**2+Y**2))
v = (np.tanh(np.sqrt(X**2+Y**2))-alpha)*np.sin(m*np.angle(X+1j*Y) - np.sqrt(X**2+Y**2))


# In[5]:


A1 = copy.deepcopy(X)
#print(np.shape(A1))


# In[6]:


A2 = copy.deepcopy(u)
#print(np.shape(A2))


# In[7]:


# initial conditions in fourier space, vector form
U_hat_0 = fft2(u)
V_hat_0 = fft2(v)   

#converting to a vector because to send to the solver
U_hat_0_vec = U_hat_0.reshape(n**2,order= 'F')
V_hat_0_vec = V_hat_0.reshape(n**2,order= 'F')


# In[8]:


A3 = copy.deepcopy(np.real(U_hat_0))
#print(np.shape(A3))


# In[9]:


init_vec = np.concatenate((U_hat_0_vec,V_hat_0_vec),axis=0).squeeze()
#print(np.shape(init_vec))


# In[10]:


#print(np.shape(init_vec[4096:8192]))


# In[11]:


#init_vec[4096:8192].reshape(n,n)


# In[12]:


A4 = copy.deepcopy(np.imag(init_vec.reshape(8192,1)))
#print(np.shape(A4))
#A4


# In[13]:


def vortrhs(t, vec, D1, D2, beta, n, KX, KY):
    # Reshape the Fourier-transformed vector
    # into an nxn matrix
    u_double_hat = vec[0:4096].reshape(n, n,order= 'F')
    v_double_hat = vec[4096:8192].reshape(n, n, order= 'F')
    
    #Write all of the terms in the physical space
    u = np.real(ifft2(u_double_hat))
    v = np.real(ifft2(v_double_hat))
    #need fourier transformed data--->u_double_hat
    
    #calculate a, lambda, omega (vectors)
    A_2 = u**2 + v**2
    lambda_A = 1 - A_2
    omega_A = -beta*A_2
    
    #print("A_2", np.shape(A_2))
    #print("lambda_A", np.shape(lambda_A))
    #print("omega_A", np.shape(omega_A))
    
    #calculate u_t and v_t
    # Ut= λ(A)U −ω(A)V + D1∇2U
    # Vt= ω(A)U −λ(A)V + D2∇2V
    
    NL_u = lambda_A*u - omega_A*v 
    NL_v = omega_A*u - lambda_A*v
    
    #print("NL_u", np.shape(NL_u))
    #print("NL_u", np.shape(NL_u))
    
    u_t = fft2(NL_u) + D1*(-(KX**2)*u_double_hat-(KY**2)*u_double_hat)
    v_t = fft2(NL_v) + D2*(-(KX**2)*v_double_hat-(KY**2)*v_double_hat)
    
    #print("u_t", np.shape(u_t))
    #print("v_t", np.shape(v_t))
    
    #Convert everything back and return a vector in the Fourier space.
    u_t_col = u_t.reshape(n**2,order= 'F')
    v_t_col = v_t.reshape(n**2,order= 'F')
    
    #print("u_t_col", np.shape(u_t_col))
    #print("v_t_col", np.shape(v_t_col))
    
    rhs = np.concatenate([u_t_col,v_t_col],axis=0).squeeze()

    return rhs


# In[14]:


#np.shape(init_vec)
#print(tspan)


# In[15]:


# integrate in fourier space
sol = scipy.integrate.solve_ivp(lambda t,wfvec: vortrhs(t, wfvec, D1, D2, beta, n, KX, KY), [0, 25], init_vec, t_eval = tspan)


# In[16]:


np.shape(sol.y)


# In[17]:


A5 = copy.deepcopy(np.real(sol.y))
#print(np.shape(A5))
#A5


# In[18]:


A6 = copy.deepcopy(np.imag(sol.y))
#print(np.shape(A6))


# In[19]:


#np.shape(sol.y[:,5])
#t=2 on the 5th


# In[20]:


#np.shape(np.real(sol.y[0:4096,5])) 


# In[21]:


#np.sqrt(4096)


# In[22]:


A7 = copy.deepcopy(np.real(sol.y[0:4096,4]).reshape(4096,1))
#print(np.shape(A7))

# u vector is in 0 to 4096
A8 = copy.deepcopy((np.real(sol.y[0:4096,4])).reshape(64,64).T)
#print(np.shape(A8))

A9 = copy.deepcopy(np.real(ifft2(sol.y[0:4096,4].reshape(64,64).T)))
#print(np.shape(A9))
#A9


# In[23]:


#print(tspan[4])


# In[24]:


#plt.scatter(x, y, c=z, cmap='jet',vmin=0, vmax=250)
#plt.colorbar()
#plt.show()


# In[25]:


#print(A9)
#np.real(sol.y[0:4096,5]).reshape(64,64)
#print(np.shape(tspan))
#print(np.shape(np.real(sol.y[0:4096,50])))
#print(np.shape(X))
#print(np.shape(Y))


# In[26]:


#np.random.seed(19680801)
#Z = np.real(sol.y[0:4096,50])

#fig, ax = plt.subplots()
#ax.pcolormesh(X, Y, Z)


# In[27]:


#np.sqrt(4096)


# In[28]:


# Generate data for the plot
#[X,Y]=np.meshgrid(x,y)
#tspan[50] =25
#z = np.real(ifft2(sol.y[0:4096,50].reshape(64,64)))
#print(np.shape(z))

# Generate the plot
#fig, ax = plt.subplots()
#cmap = ax.pcolor(X, Y, z.T)
#fig.colorbar(cmap)
#plt.show(fig)


# ## Problem 2

# Dirichlet Boundary conditions,
# U(x = −10,y,t) = U(x = 10,y,t) = U(x,y = −10,t) = U(x,y = 10,t) = 0,
# using the Chebyshev pseudo-spectral method. To do so, set m = 2, α = 1, and use
# n = 30 when creating the Chebyshev matrix and Chebyshev points. You should
# use the functions I provided for creating the matrix and points.

# In[29]:


# differentiation cheb matrix +b.c.

# We want to solve with 20 interior points. 
##N = 20
N = 30


# Create Chebyshev matrix and Chebyshev grid.
# D is an (N+1)x(N+1) matrix.
[D, x] = cheb(N)
#x = x.reshape(N+1) #get the x values from Chebyshev function

L = 20
# Rescale x
x = x*L/2 #, x goes from -10 to +10

# Derivative operator squared is second derivative
D2 = D@D

# Apply BCs
# Remove the first and last row and column
D2 = D2[1:-1, 1:-1]
#print(np.shape(D2))
# Make sure x changes accordingly too
x2 = x[1:-1]
y2 = x2.copy()
[X,Y]=np.meshgrid(x2,y2)

# This rescales the derivative
D2 = 4/(L**2)*D2 #---->how do we get this?

#tspan = np.arange(0,1+0.05,0.05)
tspan = np.arange(0,25+0.5,0.5)

#print(np.shape(tspan))
#print(np.shape(x2))
#print(np.shape(y2))
#print(np.shape(D2))


# In[30]:


# Explore two different initial conditions.
#Uinit = np.exp(-(X**2+Y**2)/0.1);
#print(np.shape(Uinit))
# Uinit = np.exp(-(X**2+Y**2)/0.1)*np.cos((X-1)*5)*np.cos((Y-1)*5)

#reshape to a vector, for solving
#uinitvec = Uinit.reshape((N-1)**2)
#print(np.shape(uinitvec))


# In[31]:


# Explore two different initial conditions.
m2 = 2
alpha2 = 1

#initial conditions
u = (np.tanh(np.sqrt(X**2+Y**2))-alpha2)*np.cos(m2*np.angle(X+1j*Y) - np.sqrt(X**2+Y**2))
v = (np.tanh(np.sqrt(X**2+Y**2))-alpha2)*np.sin(m2*np.angle(X+1j*Y) - np.sqrt(X**2+Y**2))
#print(np.shape(u))
#print(np.shape(v))


# In[32]:


u = u.T.reshape((N-1)**2)
v = v.T.reshape((N-1)**2)


# In[33]:


U_V_init = np.concatenate([u,v])
#u_v_initvec = U_V_init.T.reshape((N-1)**2)


# In[34]:


I = np.eye(len(D2))
#print(np.shape(I))
# laplacian matrix. Use "kron" to do that. Don't need to worry so much
# about how it works.
Lap = np.kron(D2,I)+np.kron(I,D2);
#print(np.shape(Lap))


# In[35]:


A10 = copy.deepcopy(Lap)
#print(np.shape(A10))


# In[36]:


A11 = copy.deepcopy(Y)
#print(np.shape(A11))


# In[37]:


A12 = copy.deepcopy(v.reshape(29,29).T)
#print(np.shape(A12))
#A12


# In[38]:


A13 = copy.deepcopy(U_V_init.reshape(1682,1))
#print(np.shape(A13))
#print(A13)


# In[39]:


D21 = 0.1
D22 = 0.1


# where A2 = U2 + V 2 and ∇2 = ∂2x+ ∂2y. We will consider a particular system with
# λ(A) = 1 −A2 (3)
# ω(A) = −βA2,

# In[40]:


def rhs_heat_cheb(t, init_vec, beta, Lap, D21, D22, N):
    #This is really simple now, just like finite difference
    #breakdown u to two separate vectors coressponding to U and V
    # into an nxn matrix
    u = init_vec[0:841]
    v = init_vec[841:]
    #print("u", np.shape(u))
    #print("v", np.shape(u))
    #print("Lap", np.shape(Lap))
    
    #calculate a, lambda, omega (vectors)
    A_2 = u**2 + v**2
    lambda_A = 1 - A_2
    omega_A = -beta*A_2
    
    #print("A_2", np.shape(A_2))
    #print("lambda_A", np.shape(lambda_A))
    #print("omega_A", np.shape(omega_A))
    
    #calculate u_t and v_t
    # Ut= λ(A)U −ω(A)V + D1∇2U
    # Vt= ω(A)U −λ(A)V + D2∇2V
    u_t = lambda_A*u - omega_A*v + D21*Lap@u
    #print("u_t", np.shape(u_t))
    v_t = omega_A*u - lambda_A*v + D22*Lap@v
    #print("v_t", np.shape(v_t))
    
    #concatenate u_t and v_t
    rhs = np.concatenate([u_t,v_t])
    
    return rhs


# In[41]:


sol2 = scipy.integrate.solve_ivp(lambda t,uvec: rhs_heat_cheb(t, uvec, beta, Lap, D21, D22, N), [0, 25+0.5], U_V_init, t_eval = tspan)
#ysol2=sol2.y


# In[42]:


#print(np.shape(sol2.y))


# In[43]:


A14 = copy.deepcopy(sol2.y.T)
#print(np.shape(A14))
#A14


# In[44]:


#tspan


# In[45]:


A15 = copy.deepcopy(sol2.y[841:1682,4].reshape(841,1))
#print(np.shape(A15))
#A15


# In[46]:


A16 = np.pad(A15.reshape(29,29).T,[1,1])
#np.shape(A16)
#A16


# In[47]:


# Generate data for the plot

# Create the grid again
#y = x.copy()
#[X,Y]=np.meshgrid(x2,y2)

# t=25 for v vector 
#z4 = np.real(sol2.y[841:1682,50]).reshape(29,29)
#print(np.shape(z4))

# Generate the plot
#fig, ax = plt.subplots()
#cmap = ax.pcolor(X, Y, z4.T)
#fig.colorbar(cmap)
#plt.show(fig)


# In[ ]:




