import numpy as np

# Lorenz parameters
sigma, rho, beta = 10.0, 28.0, 8.0/3.0

def lorenz(vec):
    x, y, z = vec
    dxdt = sigma*(y - x)
    dydt = x*(rho - z) - y
    dzdt = x*y - beta*z
    return np.array([dxdt, dydt, dzdt])

# Grid search for equilibria
x_range = np.linspace(-20, 20, 81)
y_range = np.linspace(-20, 20, 81)
z_range = np.linspace(-2, 30, 65)
threshold = 1e-2

candidates = []
for x in x_range:
    for y in y_range:
        for z in z_range:
            F = lorenz([x, y, z])
            if np.linalg.norm(F) < threshold:
                candidate = np.round([x, y, z], 3)
                if not any(np.allclose(candidate, c, atol=1e-2) for c in candidates):
                    candidates.append(candidate)

print("Equilibrium points (approximate grid):")
for c in candidates:
    print(f"x={c[0]}, y={c[1]}, z={c[2]}")

def numerical_jacobian(f, x, eps=1e-6):
    x = np.array(x)
    n = len(x)
    J = np.zeros((n, n))
    fx = np.array(f(x))
    for i in range(n):
        x1 = np.array(x)
        x1[i] += eps
        fx1 = np.array(f(x1))
        J[:, i] = (fx1 - fx) / eps
    return J

print("\nStability analysis (numerical Jacobian):")
for eq in candidates:
    J_num = numerical_jacobian(lorenz, eq)
    eigenvals = np.linalg.eigvals(J_num)
    print(f"\nAt equilibrium {eq}:")
    print(f"Jacobian:\n{J_num}")
    print(f"Eigenvalues: {eigenvals}")
    if np.all(np.real(eigenvals) < 0):
        print("Stable (attractor)")
    elif np.all(np.real(eigenvals) > 0):
        print("Unstable (repellor)")
    else:
        print("Saddle point or center (mixed stability)")