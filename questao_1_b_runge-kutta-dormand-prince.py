import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import Symbol, Function, Eq, exp, dsolve, simplify, lambdify, sin, Derivative

#============================================================================#
#                  Defining function for the linear oscilator                #
#============================================================================#
def function(x, t):
    # x[0] = x1, x[1] = x2
    dxdt = np.zeros_like(x)
    dxdt[0] = x[1]
    dxdt[1] = gamma * np.sin(omega * t) - 2 * psi * wn * x[1] - (wn ** 2) * x[0]
    return dxdt

#============================================================================#
#                    Runge-Kutta-Dormand-Prince (RK45) Integrator            #
#============================================================================#
def rk45_step(f, x, t, dt):
    # Dormand-Prince 5(4) coefficients
    c2 = 1/5
    c3 = 3/10
    c4 = 4/5
    c5 = 8/9
    c6 = 1.0
    c7 = 1.0

    a21 = 1/5

    a31 = 3/40
    a32 = 9/40

    a41 = 44/45
    a42 = -56/15
    a43 = 32/9

    a51 = 19372/6561
    a52 = -25360/2187
    a53 = 64448/6561
    a54 = -212/729

    a61 = 9017/3168
    a62 = -355/33
    a63 = 46732/5247
    a64 = 49/176
    a65 = -5103/18656

    a71 = 35/384
    a72 = 0
    a73 = 500/1113
    a74 = 125/192
    a75 = -2187/6784
    a76 = 11/84

    b1 = 35/384
    b2 = 0
    b3 = 500/1113
    b4 = 125/192
    b5 = -2187/6784
    b6 = 11/84
    b7 = 0

    k1 = f(x, t)
    k2 = f(x + dt * a21 * k1, t + c2 * dt)
    k3 = f(x + dt * (a31 * k1 + a32 * k2), t + c3 * dt)
    k4 = f(x + dt * (a41 * k1 + a42 * k2 + a43 * k3), t + c4 * dt)
    k5 = f(x + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), t + c5 * dt)
    k6 = f(x + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), t + c6 * dt)
    k7 = f(x + dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6), t + c7 * dt)

    x_next = x + dt * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7)
    return x_next

def integrate_rk45(t0, dt, n, x0, func):
    data = []
    x = np.array(x0)
    t = t0
    data.append([t] + list(x))

    for i in range(n):
        x = rk45_step(func, x, t, dt)
        t = t + dt
        data.append([t] + list(x))

    return np.array(data)

#============================================================================#
#                             MAIN PROGRAM                                   #
#============================================================================#
if __name__ == '__main__':

    # Indexes for plotting
    mt_ = 0
    mX1_ = 1
    mX2_ = 2

    #========================================================================#
    # Inputs                                                                 #
    #========================================================================#
    t0 = 0.0
    tf = 20.
    dt = 1.e-2
    Npt = int((tf-t0)/dt)
    x0 = np.array([0.0, 0.0])

    #============================================================================#
    # Constants                                                                  #
    #============================================================================#
    gamma = 0.1
    omega = 1.5
    psi = 1.35 # Overdamped system
    wn = 10.5

    #============================================================================#
    # Solution using explicit RK45                                               #
    #============================================================================#
    matrix = integrate_rk45(t0, dt, Npt, x0, function)

    #============================================================================#
    # Plots                                                                      #
    #============================================================================#
    plt.figure()
    plt.plot(matrix[:, mt_], gamma * np.sin(matrix[:, mt_]), 'red')
    plt.title("Forcing function ")
    plt.xlabel('t')
    plt.ylabel('F(t)')
    plt.grid(True)

    plt.figure()
    plt.plot(matrix[:, mt_], matrix[:, mX1_], 'blue')
    plt.title(" RK45 (Dormand-Prince) solution of linear oscillator ")
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.grid(True)

    # Analytic solution part is omitted, as per instruction (no sympy/scipy)
    # If you want, you can write the analytic solution as a function by hand if known.
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