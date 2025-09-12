import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Pendulum parameters
g, l = 9.81, 1.0

def pendulum(t, y):
    theta, omega = y
    dtheta = omega
    domega = -g/l * np.sin(theta)
    return [dtheta, domega]

# Grid search for equilibria
theta_range = np.linspace(-2*np.pi, 2*np.pi, 33)
omega_range = np.linspace(-8, 8, 9)
threshold = 1e-2
equilibria = []
for theta in theta_range:
    for omega in omega_range:
        F = np.array(pendulum(0, [theta, omega]))
        if np.linalg.norm(F) < threshold:
            candidate = np.round([theta, omega], 3)
            if not any(np.allclose(candidate, c, atol=1e-2) for c in equilibria):
                equilibria.append(candidate)
equilibria = np.array(equilibria)

def closest_equilibrium(y):
    if len(equilibria) == 0:
        return -1
    dists = [np.linalg.norm(y - eq) for eq in equilibria]
    return np.argmin(dists)

def basin_of_attraction(
        grid_size=150,
        xlim=(-2*np.pi, 2*np.pi), ylim=(-8, 8),
        tmax=30):
    X, Y = np.meshgrid(np.linspace(*xlim, grid_size), np.linspace(*ylim, grid_size))
    basin = np.full(X.shape, -1, dtype=int)
    tol = 0.15
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x0, y0 = X[i, j], Y[i, j]
            sol = solve_ivp(pendulum, [0, tmax], [x0, y0], method='RK45', rtol=1e-6, atol=1e-8)
            final = sol.y[:, -1]
            idx = closest_equilibrium(final)
            if idx >= 0 and np.linalg.norm(final - equilibria[idx]) < tol:
                basin[i, j] = idx
    plt.figure(figsize=(8,6))
    plt.imshow(basin, extent=(*xlim, *ylim), origin='lower', cmap='tab10', alpha=0.8)
    for eq in equilibria:
        plt.plot(eq[0], eq[1], 'ko')
    plt.xlabel('theta')
    plt.ylabel('omega')
    plt.title(f'Pendulum Basin of Attraction')
    plt.colorbar(ticks=range(len(equilibria)))
    plt.show()

if __name__ == '__main__':
    basin_of_attraction()