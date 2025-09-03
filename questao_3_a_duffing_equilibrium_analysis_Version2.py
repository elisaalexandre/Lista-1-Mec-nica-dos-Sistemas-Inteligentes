import numpy as np
import sympy as sp

# Duffing oscillator parameters
delta = 0.2
alpha = -1.0
beta = 1.0

# Symbolic solution for equilibria
x, y = sp.symbols('x y')
f_sym = y
g_sym = -delta * y - alpha * x - beta * x**3

# Find equilibria analytically for Duffing oscillator
equil_x = sp.solve(-alpha*x - beta*x**3, x)
equilibria = [(float(x0), 0.0) for x0 in equil_x]

print("Equilibrium points (analytic):")
for eq in equilibria:
    print(f"x = {eq[0]}, y = {eq[1]}")

# Jacobian matrix
J = sp.Matrix([
    [sp.diff(f_sym, x), sp.diff(f_sym, y)],
    [sp.diff(g_sym, x), sp.diff(g_sym, y)]
])

def analyze_stability(eq):
    subs = {x: eq[0], y: eq[1]}
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