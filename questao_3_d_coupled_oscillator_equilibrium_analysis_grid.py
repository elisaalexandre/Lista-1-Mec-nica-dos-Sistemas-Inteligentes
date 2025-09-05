import numpy as np

# System parameters
zeta1 = 0.1
zeta2 = 0.1
alpha1 = 1.0
beta1 = 1.0
alpha2 = 1.0
beta2 = 1.0
rho = 1.0
Omega_s = 1.0

def system(vec):
    x1, v1, x2, v2 = vec
    dx1dt = v1
    dv1dt = -2*zeta1*v1 + 2*zeta2*(v2-v1) - (1+alpha1)*x1 - beta1*x1**3 + rho*Omega_s**2*(x2-x1)
    dx2dt = v2
    dv2dt = (-2*zeta2*(v2-v1) - alpha2*x2 - beta2*x2**3 - rho*Omega_s**2*(x2-x1))/rho
    return np.array([dx1dt, dv1dt, dx2dt, dv2dt])

# Grid search for equilibria
x1_range = np.linspace(-2, 2, 21)
v1_range = np.linspace(-2, 2, 7)
x2_range = np.linspace(-2, 2, 21)
v2_range = np.linspace(-2, 2, 7)
threshold = 1e-2

candidates = []
for x1 in x1_range:
    for v1 in v1_range:
        for x2 in x2_range:
            for v2 in v2_range:
                F = system([x1, v1, x2, v2])
                if np.linalg.norm(F) < threshold:
                    candidate = np.round([x1, v1, x2, v2], 4)
                    if not any(np.allclose(candidate, c, atol=1e-2) for c in candidates):
                        candidates.append(candidate)

print("Equilibrium points (approximate grid):")
for c in candidates:
    print(f"x1={c[0]}, v1={c[1]}, x2={c[2]}, v2={c[3]}")

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
    J_num = numerical_jacobian(system, eq)
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