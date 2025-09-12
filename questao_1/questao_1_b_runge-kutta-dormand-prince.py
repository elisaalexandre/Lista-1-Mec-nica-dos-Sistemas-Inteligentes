# -*- coding: utf-8 -*-
"""
Runge-Kutta-Dormand-Prince (RK45) explicit implementation
No use of sympy or scipy.integrate
With variable (adaptive) step size RK45
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import Symbol, Function, Eq, exp, dsolve, simplify, lambdify, sin, Derivative

#============================================================================#
#                  Defining function for the linear oscillator                #
#============================================================================#
def function(x, t):
    # x[0] = x1, x[1] = x2
    dxdt = np.zeros_like(x)
    dxdt[0] = x[1]
    dxdt[1] = gamma * np.sin(omega * t) - 2 * psi * wn * x[1] - (wn ** 2) * x[0]
    return dxdt

#============================================================================#
#     Adaptive Runge-Kutta-Dormand-Prince (RK45) Step and Integrator         #
#============================================================================#
def rk45_step_adaptive(f, x, t, dt, atol=1e-6, rtol=1e-6):
    """
    Perform one adaptive RK45 step for dx/dt = f(x, t).
    Returns: x_next, t_next, dt_next, step_accepted
    """
    # Dormand-Prince coefficients
    c2 = 1/5; c3 = 3/10; c4 = 4/5; c5 = 8/9; c6 = 1.0; c7 = 1.0
    a21 = 1/5
    a31 = 3/40;   a32 = 9/40
    a41 = 44/45;  a42 = -56/15;   a43 = 32/9
    a51 = 19372/6561; a52 = -25360/2187; a53 = 64448/6561; a54 = -212/729
    a61 = 9017/3168;  a62 = -355/33; a63 = 46732/5247; a64 = 49/176; a65 = -5103/18656
    a71 = 35/384; a72 = 0; a73 = 500/1113; a74 = 125/192; a75 = -2187/6784; a76 = 11/84

    # b for 5th order, b_star for 4th order
    b =      np.array([35/384,      0, 500/1113, 125/192, -2187/6784,   11/84,      0])
    b_star = np.array([5179/57600,  0, 7571/16695,393/640,-92097/339200,187/2100, 1/40])
    
    k1 = f(x, t)
    k2 = f(x + dt*a21*k1, t + c2*dt)
    k3 = f(x + dt*(a31*k1 + a32*k2), t + c3*dt)
    k4 = f(x + dt*(a41*k1 + a42*k2 + a43*k3), t + c4*dt)
    k5 = f(x + dt*(a51*k1 + a52*k2 + a53*k3 + a54*k4), t + c5*dt)
    k6 = f(x + dt*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5), t + c6*dt)
    k7 = f(x + dt*(a71*k1 + a72*k2 + a73*k3 + a74*k4 + a75*k5 + a76*k6), t + c7*dt)

    # 5th order solution
    x5 = x + dt*(b[0]*k1 + b[1]*k2 + b[2]*k3 + b[3]*k4 + b[4]*k5 + b[5]*k6 + b[6]*k7)
    # 4th order solution
    x4 = x + dt*(b_star[0]*k1 + b_star[1]*k2 + b_star[2]*k3 + b_star[3]*k4 +
                 b_star[4]*k5 + b_star[5]*k6 + b_star[6]*k7)
    # Error estimate
    e = np.abs(x5 - x4)
    tol = atol + rtol * np.maximum(np.abs(x5), np.abs(x))
    err_ratio = np.max(e / tol)
    
    # Safety factor, min/max factors
    safety = 0.9
    fac_min = 0.2
    fac_max = 5.0

    # Calculate new step size
    if err_ratio == 0:
        dt_new = dt * fac_max
        accept = True
    elif err_ratio <= 1.0:
        dt_new = dt * min(fac_max, max(fac_min, safety * err_ratio ** -0.25))
        accept = True
    else:
        dt_new = dt * max(fac_min, safety * err_ratio ** -0.25)
        accept = False

    if accept:
        return x5, t + dt, dt_new, True
    else:
        return x, t, dt_new, False

def integrate_rk45_adaptive(f, x0, t0, tf, dt_init=1e-2, atol=1e-6, rtol=1e-6):
    x = np.array(x0)
    t = t0
    dt = dt_init
    T = [t]
    X = [x.copy()]
    while t < tf:
        if t + dt > tf:
            dt = tf - t
        x_new, t_new, dt_new, accept = rk45_step_adaptive(f, x, t, dt, atol, rtol)
        if accept:
            t = t_new
            x = x_new
            T.append(t)
            X.append(x.copy())
        dt = dt_new
    return np.array(T), np.array(X)

#============================================================================#
#                             MAIN PROGRAM                                   #
#============================================================================#
if __name__ == '__main__':

    # Indexes for plotting
    mX1_ = 0  # position
    mX2_ = 1  # velocity

    #========================================================================#
    # Inputs                                                                 #
    #========================================================================#
    t0 = 0.0
    tf = 20.
    dt_init = 1.e-2
    x0 = np.array([0.0, 0.0])   # x(0), dx/dt(0)

    #============================================================================#
    # Constants                                                                  #
    #============================================================================#
    gamma = 0.1
    omega = 1.5
    psi = 1.35 # Overdamped system
    wn = 10.5

    #============================================================================#
    # Solution using variable-step RK45                                          #
    #============================================================================#
    T, X = integrate_rk45_adaptive(function, x0, t0, tf, dt_init=dt_init, atol=1e-8, rtol=1e-6)

    #============================================================================#
    # Plots                                                                      #
    #============================================================================#
    plt.figure()
    plt.plot(T, gamma * np.sin(omega * T), 'red')
    plt.title("Forcing function ")
    plt.xlabel('t')
    plt.ylabel('F(t)')
    plt.grid(True)

    plt.figure()
    plt.plot(T, X[:, mX1_], 'blue')
    plt.title("Variable-step RK45 solution of linear oscillator ")
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.grid(True)

    plt.show()
    
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

t_data = T
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

error = x_data - X[:, mX1_]

plt.figure()
plt.plot(t_data, error, 'purple')
plt.title(" Error between RK45 and Analytic Solutions ")
plt.xlabel('t')
plt.ylabel('Error')
plt.grid(True)
