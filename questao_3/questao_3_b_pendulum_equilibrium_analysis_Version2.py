import numpy as np
import sympy as sp

# Pendulum parameters
g = 9.81  # gravity
l = 1.0   # length

# Symbolic vars
theta, omega = sp.symbols('theta omega')
f_sym = omega
g_sym = -g/l * sp.sin(theta)

# Analytic equilibrium points
# omega = 0, sin(theta) = 0 => theta = n*pi
n_vals = [-2, -1, 0, 1, 2]  # you can add more for more points
equilibria = [(float(n*np.pi), 0.0) for n in n_vals]

print("Equilibrium points (analytic):")
for eq in equilibria:
    print(f"theta = {eq[0]:.4f}, omega = {eq[1]:.1f}")

# Jacobian matrix
J = sp.Matrix([
    [sp.diff(f_sym, theta), sp.diff(f_sym, omega)],
    [sp.diff(g_sym, theta), sp.diff(g_sym, omega)]
])

def analyze_stability(eq):
    subs = {theta: eq[0], omega: eq[1]}
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