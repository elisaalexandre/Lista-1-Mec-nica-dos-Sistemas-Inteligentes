import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Duffing parameters
delta, alpha, beta = 0.2, -1.0, 1.0

def duffing(t, y):
    x, v = y
    dxdt = v
    dvdt = -delta*v - alpha*x - beta*x**3
    return [dxdt, dvdt]

# Grid search for equilibria
x_range = np.linspace(-2, 2, 41)
y_range = np.linspace(-2, 2, 41)
threshold = 1e-3
equilibria = []
for x in x_range:
    for y in y_range:
        F = np.array(duffing(0, [x, y]))
        if np.linalg.norm(F) < threshold:
            candidate = np.round([x, y], 4)
            if not any(np.allclose(candidate, c, atol=1e-3) for c in equilibria):
                equilibria.append(candidate)
equilibria = np.array(equilibria)

def closest_equilibrium(y):
    if len(equilibria) == 0:
        return -1
    dists = [np.linalg.norm(y - eq) for eq in equilibria]
    return np.argmin(dists)

def basin_of_attraction(
        grid_size=150,
        xlim=(-2, 2), ylim=(-2, 2),
        tmax=40):
    X, Y = np.meshgrid(np.linspace(*xlim, grid_size), np.linspace(*ylim, grid_size))
    basin = np.full(X.shape, -1, dtype=int)
    tol = 0.1
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x0, y0 = X[i, j], Y[i, j]
            sol = solve_ivp(duffing, [0, tmax], [x0, y0], method='RK45', rtol=1e-6, atol=1e-8)
            final = sol.y[:, -1]
            idx = closest_equilibrium(final)
            if idx >= 0 and np.linalg.norm(final - equilibria[idx]) < tol:
                basin[i, j] = idx
    plt.figure(figsize=(8,6))
    plt.imshow(basin, extent=(*xlim, *ylim), origin='lower', cmap='tab10', alpha=0.8)
    for eq in equilibria:
        plt.plot(eq[0], eq[1], 'ko')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Duffing Oscillator Basin of Attraction')
    plt.colorbar(ticks=range(len(equilibria)))
    plt.show()

if __name__ == '__main__':
    basin_of_attraction()