import numpy as np
import sympy as sp

# Lorenz system parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Symbolic variables
x, y, z = sp.symbols('x y z')

# Lorenz equations
f1 = sigma * (y - x)
f2 = x * (rho - z) - y
f3 = x * y - beta * z

# Analytical equilibrium points
# 1. Origin
equilibria = [(0.0, 0.0, 0.0)]

# 2. Non-trivial equilibria if rho > 1
if rho > 1:
    v = float(np.sqrt(beta * (rho - 1)))
    equilibria.append((v, v, rho - 1))
    equilibria.append((-v, -v, rho - 1))

print("Equilibrium points (analytic):")
for eq in equilibria:
    print(f"x = {eq[0]:.4f}, y = {eq[1]:.4f}, z = {eq[2]:.4f}")

# Jacobian matrix
J = sp.Matrix([
    [sp.diff(f1, x), sp.diff(f1, y), sp.diff(f1, z)],
    [sp.diff(f2, x), sp.diff(f2, y), sp.diff(f2, z)],
    [sp.diff(f3, x), sp.diff(f3, y), sp.diff(f3, z)]
])

def analyze_stability(eq):
    subs = {x: eq[0], y: eq[1], z: eq[2]}
    J_num = np.array(J.subs(subs)).astype(np.float64)
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

print("\nStability analysis:")
for eq in equilibria:
    analyze_stability(eq)