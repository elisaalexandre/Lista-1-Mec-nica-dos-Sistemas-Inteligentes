import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp

# Parameters
delta, alpha, beta = 0.2, -1.0, 1.0

def duffing(t, X):
    x, y = X
    dxdt = y
    dydt = -delta*y - alpha*x - beta*x**3
    return [dxdt, dydt]

# Equilibria: y=0, x=0 or x=±sqrt(-alpha/beta)
roots = [0.0]
if alpha/beta < 0:
    r = np.sqrt(-alpha/beta)
    roots += [r, -r]
equilibria = [(r, 0.0) for r in roots]

def closest_equilibrium(pos):
    dists = [np.linalg.norm(np.array(pos)-np.array(eq)) for eq in equilibria]
    return np.argmin(dists)

def basin_of_attraction(grid_size=300, xlim=(-2,2), ylim=(-3,3), tmax=30):
    X, Y = np.meshgrid(np.linspace(*xlim, grid_size), np.linspace(*ylim, grid_size))
    basin = np.full(X.shape, -1, dtype=int)
    tol = 0.1

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x0, y0 = X[i, j], Y[i, j]
            sol = solve_ivp(duffing, [0, tmax], [x0, y0], method='RK45', rtol=1e-7, atol=1e-8)
            final = sol.y[:, -1]
            minidx = closest_equilibrium(final)
            if np.linalg.norm(np.array(final)-np.array(equilibria[minidx])) < tol:
                basin[i, j] = minidx
    plt.figure(figsize=(8,6))
    plt.imshow(basin, extent=(*xlim, *ylim), origin='lower', cmap='tab10', alpha=0.8, aspect='auto')
    for eq in equilibria:
        plt.plot(eq[0], eq[1], 'ko')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Duffing Oscillator Basin of Attraction')
    plt.colorbar(ticks=range(len(equilibria)))
    plt.show()

if __name__ == '__main__':
    basin_of_attraction()