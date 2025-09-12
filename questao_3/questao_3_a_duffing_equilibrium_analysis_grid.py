import numpy as np

# Duffing parameters
delta, alpha, beta = 0.2, -1.0, 1.0

def duffing(vec):
    x, y = vec
    dxdt = y
    dydt = -delta*y - alpha*x - beta*x**3
    return np.array([dxdt, dydt])

# Grid search for equilibria
x_range = np.linspace(-2, 2, 81)
y_range = np.linspace(-2, 2, 81)
threshold = 1e-3

candidates = []
for x in x_range:
    for y in y_range:
        F = duffing([x, y])
        if np.linalg.norm(F) < threshold:
            candidate = np.round([x, y], 4)
            if not any(np.allclose(candidate, c, atol=1e-3) for c in candidates):
                candidates.append(candidate)

print("Equilibrium points (approximate grid):")
for c in candidates:
    print(f"x={c[0]}, y={c[1]}")

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
    J_num = numerical_jacobian(duffing, eq)
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