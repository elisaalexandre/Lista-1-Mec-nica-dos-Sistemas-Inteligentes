import numpy as np

def rkdp45_step(f, t, y, h):
    """
    Perform one step of the Runge-Kutta-Dormand-Prince 5(4) method (RKDP45).
    Returns:
    - y_next: estimated next value
    - err: error estimate
    """
    # Dormand-Prince coefficients
    a = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    b = [
        [],
        [1/5],
        [3/40, 9/40],
        [44/45, -56/15, 32/9],
        [19372/6561, -25360/2187, 64448/6561, -212/729],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    ]
    c_sol = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    c_err = [
        5179/57600, 0, 7571/16695, 393/640,
        -92097/339200, 187/2100, 1/40
    ]
    
    k = []
    k.append(f(t, y))
    for i in range(1, 7):
        yi = y.copy()
        for j in range(i):
            yi = yi + h * b[i][j] * k[j]
        k.append(f(t + a[i]*h, yi))
    
    # 5th order solution
    y_next = y.copy()
    for i in range(7):
        y_next = y_next + h * c_sol[i] * k[i]
    
    # 4th order solution (for error estimate)
    y_err = y.copy()
    for i in range(7):
        y_err = y_err + h * c_err[i] * k[i]
    err = np.abs(y_next - y_err)
    return y_next, err

def rkdp45(f, y0, t_span, tol=1e-6, max_steps=10000):
    """
    Integrate ODE y'=f(t,y) using Dormand-Prince 5(4) with adaptive error control.
    Arguments:
    - f: function f(t, y)
    - y0: initial value (array-like)
    - t_span: tuple (t0, tf)
    - tol: error tolerance
    - max_steps: maximum number of steps
    
    Returns:
    - t_points: array of time points
    - y_points: array of solution values
    """
    t0, tf = t_span
    y = np.array(y0, dtype=float)
    t = t0
    h = (tf - t0) / 100  # initial guess for step size
    t_points = [t]
    y_points = [y.copy()]
    steps = 0
    
    while t < tf and steps < max_steps:
        if t + h > tf:
            h = tf - t
        y_next, err = rkdp45_step(f, t, y, h)
        err_norm = np.max(err)
        if err_norm < tol:
            # Accept step
            t += h
            y = y_next
            t_points.append(t)
            y_points.append(y.copy())
            # Increase step size
            h = min(h * min(5, 0.9 * (tol/err_norm)**0.2), tf-t)
        else:
            # Reject step and decrease step size
            h = max(h * max(0.1, 0.9 * (tol/err_norm)**0.25), 1e-10)
        steps += 1
    
    if steps >= max_steps:
        print("Warning: Maximum number of steps reached.")
    
    return np.array(t_points), np.array(y_points)

# Forced Damped Linear Harmonic Oscillator ODE system:
# y = [x, v] where x is position, v is velocity
# dx/dt = v
# dv/dt = -2*gamma*v - omega**2 * x + F0 * np.sin(omega_f * t)

if __name__ == "__main__":
    omega = 2.0 * np.pi   # natural angular frequency, 1 Hz
    gamma = 0.5           # damping coefficient
    F0 = 1.0              # forcing amplitude
    omega_f = 2.0 * np.pi # forcing angular frequency, 1 Hz (can be different from omega)

    def forced_damped_lho_sin(t, y):
        x, v = y
        dxdt = v
        dvdt = -2 * gamma * v - omega**2 * x + F0 * np.sin(omega_f * t)
        return np.array([dxdt, dvdt])

    y0 = [1.0, 0.0]  # initial position 1.0, initial velocity 0.0
    t_span = (0, 100)  # simulate for 100 seconds

    t_points, y_points = rkdp45(forced_damped_lho_sin, y0, t_span, tol=1e-6)

    # Print results
    for t, y in zip(t_points, y_points):
        print(f"t={t:.4f}, x={y[0]:.6f}, v={y[1]:.6f}")