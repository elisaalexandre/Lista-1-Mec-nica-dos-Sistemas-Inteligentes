# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 16:49:03 2025

@author: melis
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from sympy import Symbol, Function, Eq, exp, dsolve, simplify, lambdify, sin, Derivative


'Part 1 - Solving the problem by the Runge-Kutta 4th Order Method'


#============================================================================#
#                                 RK4 Method                                 #
#============================================================================#
def RK4(func, x0, dt, t0):
    x0 = np.asarray(x0)
    k1 = func(x0, t0)
    t = t0 + dt/2.0
    x = x0 + k1*dt/2.0
    k2 = func(x, t)
    t = t0 + dt/2.0
    x = x0 + k2*dt/2.0
    k3 = func(x, t)
    t = t0 + dt
    x = x0 + k3*dt
    k4 = func(x, t)

    return x0 + dt*(k1 + 2.0*(k2 + k3) + k4)/6.0

#============================================================================#
#                           Integration Function                             #
#============================================================================#
def integrate(t0, dt, n, x0, func):
    data = []
    x = x0
    t = t0
    data.append([t] + list(x) ) #+ [D])

    for i in range(n):
        x = RK4(func, x, dt, t)
        t = t + dt
        data.append([t] + list(x)) # + [D])

    return np.array(data)

#============================================================================#
#                  Defining function for the linear oscilator                #
#============================================================================#
def function(x, t):
    
    # global D
    
    ff = np.zeros(len(x))
          
    ff[X1_] = x[X2_] 
    ff[X2_] = gamma*np.sin(omega*t) -2*psi*wn*x[X2_] - (wn**2)*x[X1_]
       
    return ff

#============================================================================#
#                             MAIN PROGRAM                                   #
#============================================================================#
if __name__ == '__main__':
    
    #global D
    
# Index Linear Harmonic Oscillator
    X1_ = 0   
    X2_ = 1 
        
 
# Matrix index
    mt_ = 0    
    mX1_ = 1
    mX2_ = 2
    
#========================================================================#
# Inputs                                                                 #
#========================================================================#    
    t0 = 0.0
    tf= 20.
    dt = 1.e-2
    Npt = int((tf-t0)/dt)
    x0 = np.array([0.0, 0.0])
    
#============================================================================#
# Constants                                                                  #
#============================================================================#    
 
    gamma = 0.1
    omega = 1.5
    psi = 1.35 #Overdamped system
    wn = 10.5
    
#============================================================================#
# Solution                                                                   #
#============================================================================#

    
    matrix = integrate(t0, dt, Npt, x0, function)
    
#============================================================================#
# Plots                                                                      #
#============================================================================#

    plt.figure()
    plt.plot(matrix[:, mt_], gamma*np.sin(matrix[:, mt_]),'red')
    
    plt.figure()
    plt.plot(matrix[:, mt_], matrix[:, mX1_],'blue')
    plt.title(" RK4 solution of linear oscilator ")
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.grid(True)

'Part 2 - Analytic Solution'

#============================================================================#
#         Defining the independent variable and the dependent function       #
#============================================================================#

t = Symbol('t')
x = Function('x')

#============================================================================#
#       Defining the ODE: x'' + 2*psi*wn*x'+ wn**2*x = gamma*sen(omega*t)    #
#============================================================================#

ode = Eq(Derivative(x(t), t, 2) + 2*psi*wn*(Derivative(x(t), t, 1)) + (wn**2)*x(t), gamma*sin(omega*t))

#============================================================================#
#           Solving the ODE with an initial condition                        #
#============================================================================#

sol = dsolve(ode, ics={x(0): 0, x(t).diff(t).subs(t, 0): 0})

# Simplifying the solution
sol_simpl = simplify(sol.rhs)

# Converting the symbolic solution to a NumPy-callable function
f = lambdify(t, sol_simpl, 'numpy')

#============================================================================#
#                      Generating data for plotting                          #
#============================================================================#

t_data = matrix[:, mt_]
x_data = f(t_data)

# Plotting the solution
plt.figure()
plt.plot(t_data, x_data, 'green')
plt.title(" Analytic solution of linear oscilator ")
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid(True)

#============================================================================#
#                               Error                                        #
#============================================================================#

error = x_data - matrix[:, mX1_]

plt.figure()
plt.plot(t_data, error, 'purple')
plt.title(" Error between RK4 and Analytic Solutions ")
plt.xlabel('t')
plt.ylabel('Error')
plt.grid(True)
