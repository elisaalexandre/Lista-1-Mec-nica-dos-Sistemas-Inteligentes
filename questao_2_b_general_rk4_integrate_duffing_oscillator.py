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

# Duffing oscillator parameters
#delta = 0.2       # Damping
#alpha = -1.0      # Linear stiffness
#beta = 1.0        # Nonlinear stiffness
#F = 0.3           # Driving amplitude
#omega_drive = 1.2 # Driving frequency

delta = 0.2
alpha = -1.0
beta = 1.0
F = 0.4              # <-- increase to 0.5 or higher for more chaos
omega_drive = 1.2
state0 = [0.5, 0.0]
t_max = 2000 

def duffing_oscillator_ode(state, t):
    x, v = state
    dxdt = v
    dvdt = -delta*v - alpha*x - beta*x**3 + F*np.cos(omega_drive*t)
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
duffing_transform = lambda t, state: [state[0], state[1]]
sample_action = partial(sample_action_general, transform=duffing_transform)

# Initial conditions and integration parameters
state0 = [0.5, 0.0]
t0 = 0.0
dt = 0.01
#t_max = 500 * period  # Increase for more points

samples = integrate(
    duffing_oscillator_ode,
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
plt.scatter(samples[:,0], samples[:,1], s=40, color='teal')
plt.xlabel('x')
plt.ylabel('v')
plt.title('Poincaré Map (Duffing Oscillator, RK4)')
plt.grid(True)
plt.show()