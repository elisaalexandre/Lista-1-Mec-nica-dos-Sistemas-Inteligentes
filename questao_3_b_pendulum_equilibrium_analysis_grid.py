import numpy as np

# Pendulum parameters
g, l = 9.81, 1.0

def pendulum(vec):
    theta, omega = vec
    dtheta = omega
    domega = -g/l * np.sin(theta)
    return np.array([dtheta, domega])

# Grid search for equilibria
theta_range = np.linspace(-2*np.pi, 2*np.pi, 161)
omega_range = np.linspace(-8, 8, 33)
threshold = 1e-2

candidates = []
for theta in theta_range:
    for omega in omega_range:
        F = pendulum([theta, omega])
        if np.linalg.norm(F) < threshold:
            candidate = np.round([theta, omega], 4)
            if not any(np.allclose(candidate, c, atol=1e-2) for c in candidates):
                candidates.append(candidate)

print("Equilibrium points (approximate grid):")
for c in candidates:
    print(f"theta={c[0]}, omega={c[1]}")

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
    J_num = numerical_jacobian(pendulum, eq)
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