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

# Forced oscillator parameters
omega0 = 0.5           # Natural frequency
F = 0.5                # Driving amplitude
omega_drive = 9.2      # Driving frequency

def forced_oscillator_ode(state, t):
    x, v = state
    dxdt = v
    dvdt = -omega0**2 * x + F * np.cos(omega_drive * t)
    return np.array([dxdt, dvdt])

# Sample every drive period (Poincaré section)
period = 2 * np.pi / omega_drive
def period_condition_factory(period):
    next_cross = [period]
    def condition(t, state):
        if t >= next_cross[0]:
            next_cross[0] += period
            return True
        return False
    return condition

# Example transform: just (x, v)
oscillator_transform = lambda t, state: [state[0], state[1]]
sample_action = partial(sample_action_general, transform=oscillator_transform)

# Initial conditions and integration parameters
state0 = [1.0, 0.0]  # Start at x=1, v=0
t0 = 0.0
dt = 0.01
t_max = 400   # many drive cycles

samples = integrate(
    forced_oscillator_ode,
    state0,
    t0,
    dt,
    t_max,
    sample_condition=period_condition_factory(period),
    sample_action=sample_action,
    samples=[]
)

samples = np.array(samples)
plt.figure(figsize=(7, 5))
plt.scatter(samples[:,0], samples[:,1], s=40, color='purple')
plt.xlabel('x')
plt.ylabel('v')
plt.title('Poincaré Map (Forced Oscillator, RK4)')
plt.grid(True)
plt.show()