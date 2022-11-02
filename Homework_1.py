# %%
"""
# AMATH Scientific Computing
## Homework-1
### Manjaree Binjolkar
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy
from copy import deepcopy
#%matplotlib inline

# %%
# Problem 1 part a

# ODE and time steps:
y0 = np.pi/np.sqrt(2)#initial condition
f = lambda t, y: -3*y*np.sin(t)# dydt = f(t,y)

#Given exact solution:
f_exact = lambda t: (np.pi*(np.exp(3*(np.cos(t)-1))))/np.sqrt(2)

# %%
#Create the FE function
def ForwardEuler(f, y0, tspan, dt):
    error = 0
    s = np.zeros(len(tspan))#initialising
    s[0] = y0
    
    for i in range(0,len(tspan)-1):
        s[i+1] = s[i] + dt*f(tspan[i],s[i])
        #print(s[i])
    
    return s

# %%
#Create the Heun's method function
def Heuns(f, y0, tspan, dt):
    error = 0
    s = np.zeros(len(tspan))#initialising
    s[0] = y0
    
    for i in range(0,len(tspan)-1):
        
        s[i+1] = s[i] + ((dt/2)*f(tspan[i],s[i])) + f(tspan[i+1],s[i]+dt*(f(tspan[i],s[i])))
        #print(s[i])
    
    return s

# %%
#Create the RK2 function
def RK2(f, y0, tspan, dt):
    error = 0
    s = np.zeros(len(tspan))#initialising
    s[0] = y0
    
    for i in range(0,len(tspan)-1):
        s[i+1] = s[i] + dt*f(tspan[i]+(dt/2),s[i]+(dt/2)*f(tspan[i],s[i]))
        #print(s[i])
    
    return s[i+1]

# %%
#Create the FE function
def AdamsP(f, y0, tspan, dt):
    error = 0
    s = np.zeros(len(tspan))#initialising
    s[0] = y0
    s[1] =  RK2(f, s[0], tspan, dt)#calling rk2
    
    for i in range(1,len(tspan)-1):
        s_p = s[i] + (dt/2)*(3*f(tspan[i],s[i])-f(tspan[i-1],s[i-1]))
        s[i+1] = s[i] + (dt/2)*(f(tspan[i+1],s_p)+f(tspan[i],s[i]))
        #print(s[i])
    
    return s

# %%
#1st span
dt = np.power(2.,-2)
#print(dt)
tspan = np.arange(0.,5.+dt,np.power(2.,-2))
#print(tspan)

#Forward Euler method
FE_1= ForwardEuler(f, y0, tspan, dt)
#print(FE_1)
#calculating the error with respect to f_exact
#print(f_exact(5))
E1 = abs(f_exact(5)-FE_1[len(tspan)-1])
#print(E1)


#Heun's method
Heun_1 = Heuns(f, y0, tspan, dt)
#print(Heun_1)
#calculating the error with respect to f_exact
HE1 = abs(f_exact(5)-Heun_1[len(tspan)-1])
#print(HE1)


#adam method
A_1 = AdamsP(f, y0, tspan, dt)
#print(A_1)
#calculating the error with respect to f_exact
AE1 = abs(f_exact(5)-A_1[len(tspan)-1])
#print(AE1)

# %%
#2nd span
dt = np.power(2.,-3)
#print(dt)
tspan = np.arange(0.,5.+dt,np.power(2.,-3))
#print(tspan)

# Forward Euler method
FE_2= ForwardEuler(f, y0, tspan, dt)
#print(FE_1)
#calculating the error with respect to f_exact
#print(f_exact(5))
E2 = abs(f_exact(5)-FE_2[len(tspan)-1])
#print(E2)


#Heun's method
Heun_2 = Heuns(f, y0, tspan, dt)
#print(Heun_2)
#calculating the error with respect to f_exact
HE2 = abs(f_exact(5)-Heun_2[len(tspan)-1])
#print(f_exact(5))
#print(HE2)


#adam method
A_2 = AdamsP(f, y0, tspan, dt)
#print(A_2)
#calculating the error with respect to f_exact
AE2 = abs(f_exact(5)-A_2[len(tspan)-1])
#print(AE2)

# %%
#3rd span
dt = np.power(2.,-4)
#print(dt)
tspan = np.arange(0.,5.+dt,np.power(2.,-4))
#print(tspan)


#Forward Euler method
FE_3= ForwardEuler(f, y0, tspan, dt)
#print(FE_1)
#calculating the error with respect to f_exact
#print(f_exact(5))
E3 = abs(f_exact(5)-FE_3[len(tspan)-1])
#print(E3)


#Heun's method
Heun_3 = Heuns(f, y0, tspan, dt)
#print(Heun_3)
#calculating the error with respect to f_exact
HE3 = abs(f_exact(5)-Heun_3[len(tspan)-1])
#print(HE3)


#adam method
A_3 = AdamsP(f, y0, tspan, dt)
#print(A_3)
#calculating the error with respect to f_exact
AE3 = abs(f_exact(5)-A_3[len(tspan)-1])
#print(AE3)

# %%
#4th span
dt = np.power(2.,-5)
#print(dt)
tspan = np.arange(0.,5.+dt,np.power(2.,-5))
#print(tspan)


#Forward euler method
FE_4= ForwardEuler(f, y0, tspan, dt)
#print(FE_1)
#calculating the error with respect to f_exact
#print(f_exact(5))
E4 = abs(f_exact(5)-FE_4[len(tspan)-1])
#print(E4)


#Heun's method
Heun_4 = Heuns(f, y0, tspan, dt)
#print(Heun_4)
#calculating the error with respect to f_exact
HE4 = abs(f_exact(5)-Heun_4[len(tspan)-1])
#print(HE4)


#adam method
A_4 = AdamsP(f, y0, tspan, dt)
#print(A_4)
#calculating the error with respect to f_exact
AE4 = abs(f_exact(5)-A_4[len(tspan)-1])
#print(AE4)

# %%
#5th span
dt = np.power(2.,-6)
#print(dt)
tspan = np.arange(0.,5.+dt,np.power(2.,-6))
#print(tspan)


#Forward Euler method
FE_5= ForwardEuler(f, y0, tspan, dt)
#print(FE_1)
#calculating the error with respect to f_exact
#print(f_exact(5))
E5 = abs(f_exact(5)-FE_5[len(tspan)-1])
#print(E5)


#Heun's method
Heun_5 = Heuns(f, y0, tspan, dt)
#print(Heun_5)
#calculating the error with respect to f_exact
HE5 = abs(f_exact(5)-Heun_5[len(tspan)-1])
#print(HE5)


#adam method
A_5 = AdamsP(f, y0, tspan, dt)
#print(A_5)
#calculating the error with respect to f_exact
AE5 = abs(f_exact(5)-A_5[len(tspan)-1])
#print(AE5)

# %%
#6th span
dt = np.power(2.,-7)
#print(dt)
tspan = np.arange(0.,5.+dt,np.power(2.,-7))
#print(tspan)


#Forward Euler method
FE_6= ForwardEuler(f, y0, tspan, dt)
#print(FE_1)
#calculating the error with respect to f_exact
#print(f_exact(5))
E6 = abs(f_exact(5)-FE_6[len(tspan)-1])
#print(E6)


#Heun's method
Heun_6 = Heuns(f, y0, tspan, dt)
#print(Heun_6)
#calculating the error with respect to f_exact
HE6 = abs(f_exact(5)-Heun_6[len(tspan)-1])
#print(HE6)


#adam method
A_6 = AdamsP(f, y0, tspan, dt)
#print(A_6)
#calculating the error with respect to f_exact
AE6 = abs(f_exact(5)-A_6[len(tspan)-1])
#print(AE6)

# %%
#7th span
dt = np.power(2.,-8)
#print(dt)
tspan = np.arange(0.,5.+dt,np.power(2.,-8))
#print(tspan)

#Forward Euler
FE_7= ForwardEuler(f, y0, tspan, dt)
#print(FE_1)
#calculating the error with respect to f_exact
#print(f_exact(5))
E7 = abs(f_exact(5)-FE_7[len(tspan)-1])
#print(E7)


#Heun's method
Heun_7 = Heuns(f, y0, tspan, dt)
#print(Heun_7)
#calculating the error with respect to f_exact
HE7 = abs(f_exact(5)-Heun_7[len(tspan)-1])
#print(HE7)


#adam method
A_7 = AdamsP(f, y0, tspan, dt)
#print(A_1)
#calculating the error with respect to f_exact
AE7 = abs(f_exact(5)-A_7[len(tspan)-1])
#print(AE7)

# %%
A1 = deepcopy(FE_7)#2^(-8) solution for FE method
A4 = deepcopy(Heun_7)#2^(-8) solution for Heun's method
A7 = deepcopy(A_7)
#print(A4)

# %%
#Collecting all the Es for FE method
E = [E1, E2, E3, E4, E5, E6, E7]
err_FE = np.array(E)
A2 = deepcopy(err_FE)
#print(A2)

# %%
#Collecting all the Es for heuns method
HE = [HE1, HE2, HE3, HE4, HE5, HE6, HE7]
err_heun = np.array(HE)
A5 = deepcopy(err_heun)
#print(A5)

# %%
#Collecting all the Es for adam method
AE = [AE1, AE2, AE3, AE4, AE5, AE6, AE7]
err_adam = np.array(AE)
A8 = deepcopy(err_adam)
#print(A5)

# %%
delta_t_list = [np.power(2.,-2),np.power(2.,-3), np.power(2.,-4), np.power(2.,-5), np.power(2.,-6),np.power(2.,-7),np.power(2.,-8)]
delta_t = np.array(delta_t_list)
#delta_t

# %%
#Calculating the slope for FEmethod
# fit log(y) = m*log(x) + c, poly deg=1
m_fe, c_fe = np.polyfit(np.log(delta_t), np.log(err_FE),1) 
A3 = deepcopy(m_fe)
#print(A3)
# calculating the fitted values of y
y_fit_fe = np.exp(m_fe*np.log(delta_t) + c_fe) 

# %%
#Calculating the slope for heun method
# fit log(y) = m*log(x) + c, poly deg=1
m_heun, c_heun = np.polyfit(np.log(delta_t), np.log(err_heun),1) 
A6 = deepcopy(m_heun)
#print(A6)
# calculating the fitted values of y
y_fit_heun = np.exp(m_heun*np.log(delta_t) + c_heun) 

# %%
#Calculating the slope for adams method
# fit log(y) = m*log(x) + c, poly deg=1
m_a, c_a = np.polyfit(np.log(delta_t), np.log(err_adam),1) 
A9 = deepcopy(m_a)
#print(A9)
# calculating the fitted values of y
y_fit_a = np.exp(m_a*np.log(delta_t) + c_a) 

# %%
#Problem 2 part a
#Given:
t0 = 0
tf = 32
y0 = np.sqrt(3)
x0 = 1
dt = 0.5

#define the vandepols function
def vdp(t, z):
    x, y = z
    return [y, mu*(1 - x**2)*y - x]

mus = [0.1, 1, 20]
tspan = np.arange(t0,tf+dt,dt)
temp=[]
for mu in mus:
    sol = scipy.integrate.solve_ivp(vdp, [t0, tf], [y0, x0], t_eval=tspan)
    temp.append(sol.y[0])
#print(temp)

y1=temp[0:1]
y2=temp[1:2]
y3=temp[2:3]

A10_temp = [y1, y2, y3]
A10 = deepcopy(np.transpose(A10_temp))
#A10

# %%
#Problem 2 part b

#Given
y0 = 2
x0 = (np.pi)**2
mu = 1

tols = [np.power(10.,-n) for n in np.arange(4,11,1)]
#print(tols)

#initialising t_avg
t_avgs= []

# RK45
for tol in tols:
        sol = scipy.integrate.solve_ivp(vdp, [0,32], [y0, x0 ], atol=tol, rtol=tol)
        T = sol.t
        Y = sol.y
        #print(T)
        t_avgs.append(np.mean(np.diff(T)))
        
        
#print(t_avgs)  

x_input1 =np.log(t_avgs)
y_input1 =np.log(tols)


#trying to find the best fit line
s1, i1 = np.polyfit(x_input1, y_input1, 1)
A11 = deepcopy(s1)
#print(A11)


# RK23

#initialising t_avg
t_avgs2= []

for tol in tols:
        sol = scipy.integrate.solve_ivp(vdp, [0,32], [y0, x0 ], atol=tol, rtol=tol, method='RK23')
        T = sol.t
        Y = sol.y
        #print(T)
        t_avgs2.append(np.mean(np.diff(T)))

x_input2 =np.log(t_avgs2)
y_input2 =np.log(tols)

#trying to find the best fit line
s2, i2 = np.polyfit(x_input2, y_input2, 1)
A12 = deepcopy(s2)
#print(A12)



# BDF
#initialising t_avg
t_avgs3= []


for tol in tols:
        sol = scipy.integrate.solve_ivp(vdp, [0,32], [y0, x0 ], atol=tol, rtol=tol, method='BDF')
        T = sol.t
        Y = sol.y
        #print(T)
        t_avgs3.append(np.mean(np.diff(T)))
        
x_input3 =np.log(t_avgs3)
y_input3 =np.log(tols)

#trying to find the best fit line
s3, i3 = np.polyfit(x_input3, y_input3, 1)
A13 = deepcopy(s3)
#print(A13)

# %%
#Problem 3
def sys_rhs(t, y, d12, d21):
    a1 = .05; a2 = .25; b = .01; c = .01; I = .1
    v1,v2,w1,w2 = y#v1,v2,w1,w2
    dv1 = -v1 ** 3 + (1 + a1) * (v1 ** 2) - a1 * v1 - w1 + I + d12*v2
    dw1 = b * v1 - c * w1
    dv2 =- v2 ** 3 + (1 + a2) * (v2 ** 2) - a2 * v2 - w2 + I + d21*v1
    dw2=b * v2 - c * w2
    
    dydt1 = [dv1,dw1]
    dydt2 = [dv2,dw2]
    return  [dv1, dv2, dw1, dw2]#v1,v2,w1,w2

# %%
tspan = np.arange(0, 100, 0.5)

v_in=0
w_in=0

yt0 = [.1, .1, 0, 0]#v1,v2,w1,w2

d12 = 0
d21 = 0
sol_0_0 = scipy.integrate.solve_ivp(sys_rhs, tspan, y0 = yt0, args=(d12, d21), method='BDF')
A14 = deepcopy(np.transpose(sol_0_0.y))
#print(A14)

# %%
d12 = 0
d21 = 0.2
sol_0_02 = scipy.integrate.solve_ivp(sys_rhs, tspan, y0 = yt0, args=(d12, d21), method='BDF')
A15 = deepcopy(np.transpose(sol_0_02.y))
#print(A15)

# %%
d12 = -0.1
d21 = 0.2
sol_01_02 = scipy.integrate.solve_ivp(sys_rhs, tspan, y0 = yt0, args=(d12, d21), method='BDF')
A16 = deepcopy(np.transpose(sol_01_02.y))
#print(A16)

# %%
d12 = -0.3
d21 = 0.2
sol_03_02 = scipy.integrate.solve_ivp(sys_rhs, tspan, y0 = yt0, args=(d12, d21), method='BDF')
A17 = deepcopy(np.transpose(sol_03_02.y))
#print(A17)

# %%
d12 = -0.5
d21 = 0.2
sol_05_02 = scipy.integrate.solve_ivp(sys_rhs, tspan, y0 = yt0, args=(d12, d21), method='BDF')
A18 = deepcopy(np.transpose(sol_05_02.y))
#print(A18)