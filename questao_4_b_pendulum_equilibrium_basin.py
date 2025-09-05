import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp

# Parameters
g, l = 9.81, 1.0

def pendulum(t, X):
    theta, omega = X
    return [omega, -g/l * np.sin(theta)]

# Equilibria: theta = n*pi, omega = 0
n_vals = [-1, 0, 1]
equilibria = [(n*np.pi, 0.0) for n in n_vals]

def closest_equilibrium(pos):
    dists = [np.linalg.norm([((pos[0]-eq[0]+np.pi)%(2*np.pi)-np.pi), pos[1]-eq[1]]) for eq in equilibria]
    return np.argmin(dists)

def basin_of_attraction(grid_size=300, xlim=(-2*np.pi, 2*np.pi), ylim=(-8, 8), tmax=20):
    X, Y = np.meshgrid(np.linspace(*xlim, grid_size), np.linspace(*ylim, grid_size))
    basin = np.full(X.shape, -1, dtype=int)
    tol = 0.2

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x0, y0 = X[i, j], Y[i, j]
            sol = solve_ivp(pendulum, [0, tmax], [x0, y0], method='RK45', rtol=1e-7, atol=1e-8)
            final = sol.y[:, -1]
            minidx = closest_equilibrium(final)
            if np.linalg.norm([((final[0]-equilibria[minidx][0]+np.pi)%(2*np.pi)-np.pi), final[1]-equilibria[minidx][1]]) < tol:
                basin[i, j] = minidx
    plt.figure(figsize=(8,6))
    plt.imshow(basin, extent=(*xlim, *ylim), origin='lower', cmap='tab10', alpha=0.8, aspect='auto')
    for eq in equilibria:
        plt.plot(eq[0], eq[1], 'ko')
    plt.xlabel('theta')
    plt.ylabel('omega')
    plt.title('Pendulum Basin of Attraction')
    plt.colorbar(ticks=range(len(equilibria)))
    plt.show()

if __name__ == '__main__':
    basin_of_attraction()