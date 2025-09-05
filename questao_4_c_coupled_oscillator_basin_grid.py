import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# System parameters
zeta1 = 0.1
zeta2 = 0.1
alpha1 = 1.0
beta1 = 1.0
alpha2 = 1.0
beta2 = 1.0
rho = 1.0
Omega_s = 1.0

def system(t, y):
    x1, v1, x2, v2 = y
    dx1dt = v1
    dv1dt = -2*zeta1*v1 + 2*zeta2*(v2-v1) - (1+alpha1)*x1 - beta1*x1**3 + rho*Omega_s**2*(x2-x1)
    dx2dt = v2
    dv2dt = (-2*zeta2*(v2-v1) - alpha2*x2 - beta2*x2**3 - rho*Omega_s**2*(x2-x1))/rho
    return [dx1dt, dv1dt, dx2dt, dv2dt]

# Find equilibria via grid search (same as above, but coarser for speed)
x1_range = np.linspace(-2, 2, 9)
v1_range = np.linspace(-2, 2, 3)
x2_range = np.linspace(-2, 2, 9)
v2_range = np.linspace(-2, 2, 3)
threshold = 1e-2
equilibria = []
for x1 in x1_range:
    for v1 in v1_range:
        for x2 in x2_range:
            for v2 in v2_range:
                F = np.array(system(0, [x1, v1, x2, v2]))
                if np.linalg.norm(F) < threshold:
                    candidate = np.round([x1, v1, x2, v2], 4)
                    if not any(np.allclose(candidate, c, atol=1e-2) for c in equilibria):
                        equilibria.append(candidate)
equilibria = np.array(equilibria)

def closest_equilibrium(y):
    if len(equilibria) == 0:
        return -1
    dists = [np.linalg.norm(y - eq) for eq in equilibria]
    return np.argmin(dists)

def basin_of_attraction(
        grid_size=100,
        x1lim=(-2, 2), v1lim=(-2, 2),
        x2_init=0.0, v2_init=0.0,
        tmax=80):
    X1, V1 = np.meshgrid(np.linspace(*x1lim, grid_size), np.linspace(*v1lim, grid_size))
    basin = np.full(X1.shape, -1, dtype=int)
    tol = 0.1

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            y0 = [X1[i,j], V1[i,j], x2_init, v2_init]
            sol = solve_ivp(system, [0, tmax], y0, method='RK45', rtol=1e-7, atol=1e-8)
            final = sol.y[:, -1]
            idx = closest_equilibrium(final)
            if idx >= 0 and np.linalg.norm(final - equilibria[idx]) < tol:
                basin[i, j] = idx
    plt.figure(figsize=(8,6))
    plt.imshow(basin, extent=(*x1lim, *v1lim), origin='lower', cmap='tab10', alpha=0.8, aspect='auto')
    for eq in equilibria:
        plt.plot(eq[0], eq[1], 'ko')
    plt.xlabel('x1')
    plt.ylabel('v1')
    plt.title('Coupled Oscillator Basin of Attraction\n(x2,v2) initial = ({},{})'.format(x2_init, v2_init))
    plt.colorbar(ticks=range(len(equilibria)))
    plt.show()

if __name__ == '__main__':
    basin_of_attraction()