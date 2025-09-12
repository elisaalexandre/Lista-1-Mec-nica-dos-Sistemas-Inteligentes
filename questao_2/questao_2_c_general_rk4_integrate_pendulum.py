import numpy as np
import matplotlib.pyplot as plt
from functools import partial

def rk4_step(f, state, t, dt):
    k1 = np.array(f(state, t))
    k2 = np.array(f(state + 0.5*dt*k1, t + 0.5*dt))
    k3 = np.array(f(state + 0.5*dt*k2, t + 0.5*dt))
    k4 = np.array(f(state + dt*k3, t + dt))
    new_state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    new_t = t + dt
    return new_state, new_t

def integrate(f, state0, t0, dt, t_max, sample_condition=None, sample_action=None, samples=None):
    t = t0
    state = np.array(state0)
    n_steps = int((t_max - t0)/dt)
    if samples is None:
        samples = []
    for _ in range(n_steps):
        state, t = rk4_step(f, state, t, dt)
        if sample_condition is not None and sample_condition(t, state):
            if sample_action is not None:
                sample_action(t, state, samples)
            else:
                samples.append((t, state.copy()))
    return samples

def sample_action_general(t, state, samples, transform=lambda t, state: (t, state.copy())):
    samples.append(transform(t, state))

# Driven damped pendulum parameters
gamma = 0.2          # Damping
omega_d = 2/3        # Driving frequency
F_d = 1.2            # Driving amplitude

def pendulum_ode(state, t):
    theta, omega = state
    dtheta_dt = omega
    domega_dt = -gamma * omega - np.sin(theta) + F_d * np.cos(omega_d * t)
    return np.array([dtheta_dt, domega_dt])

# Sample every drive period (Poincaré section)
period = 2 * np.pi / omega_d
def period_condition_factory(period):
    next_cross = [period]
    def condition(t, state):
        if t >= next_cross[0]:
            next_cross[0] += period
            return True
        return False
    return condition

# Transform for Poincaré map: (theta mod 2pi, omega)
pendulum_transform = lambda t, state: [state[0] % (2*np.pi), state[1]]
sample_action = partial(sample_action_general, transform=pendulum_transform)

# Initial conditions and integration parameters
state0 = [0.2, 0.0]  # (theta, omega)
t0 = 0.0
dt = 0.01
t_max = 10000 

# Integrate and collect samples
samples = integrate(
    pendulum_ode,
    state0,
    t0,
    dt,
    t_max,
    sample_condition=period_condition_factory(period),
    sample_action=sample_action,
    samples=[]
)

samples = np.array(samples)
plt.figure(figsize=(8, 6))
plt.scatter(samples[:,0], samples[:,1], s=10, color='blue', alpha=0.7)
plt.xlabel('Theta (mod 2π)')
plt.ylabel('Omega')
plt.title('Poincaré Map (Driven Damped Pendulum, RK4)')
plt.grid(True)
plt.show()