import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = len(resource_grid)
    Z = len(income)
    m_grid = resource_grid

    # Work in log-space for smoother representation
    log_m_fine = np.linspace(np.log(0.001), np.log(40), 801)
    m_fine = np.exp(log_m_fine)
    n_fine = len(m_fine)

    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c**(1 - gamma) / (1 - gamma)

    def utility_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c**(-gamma)

    # Storage: c_policy_log[t][z] is a 1D array of log(consumption) at log_m_fine
    # We store log(c) as function of log(m) for smoothness
    c_policy_log = np.zeros((horizon, Z, n_fine))
    V = np.zeros((horizon, Z, n_fine))

    # Terminal: consume all -> log(c) = log(m)
    for z in range(Z):
        c_policy_log[horizon - 1, z, :] = log_m_fine.copy()
        V[horizon - 1, z, :] = utility(m_fine)

    # Pre-compute splines for the terminal period
    # We'll maintain splines for c_policy_log[t+1][z] as function of log_m_fine

    for t in range(horizon - 2, -1, -1):
        # Build splines for next period's policies
        next_splines = []
        for z in range(Z):
            # log_c as function of log_m
            cs = CubicSpline(log_m_fine, c_policy_log[t + 1, z, :], bc_type='not-a-knot')
            next_splines.append(cs)

        for z in range(Z):
            # Initial guess: proportional consumption in log-space
            # log(c) = log(m) - log(2) roughly, but let's use a simple fraction
            c_init = 0.5 * m_fine
            c_init = np.clip(c_init, 1e-12, m_fine * (1 - 1e-10))
            log_c = np.log(c_init)

            # Fixed-point iteration on Euler equation in log-space
            # u'(c) = beta * R * sum_z' P[z,z'] * u'(c'(m', z'))
            # In log-space: -gamma * log(c) = log(beta*R) + log(sum P * u'(c'))  [for gamma != 1]
            # For gamma=1: -log(c) = log(beta*R) + log(sum P * u'(c'))
            # General: u'(c) = c^(-gamma) for gamma!=1, 1/c for gamma=1
            # So: c^(-gamma) = beta*R*sum P*u'(c')
            # => -gamma * log(c) = log(beta*R) + log(sum P * u'(c'))
            # => log(c) = -(1/gamma) * (log(beta*R) + log(sum P * u'(c')))  for gamma != 1
            # For gamma=1: log(c) = -(log(beta*R) + log(sum P * u'(c')))

            for fp_iter in range(50):
                log_c_new = np.zeros(n_fine)
                for i in range(n_fine):
                    m_val = m_fine[i]
                    c_val = np.exp(log_c[i])
                    c_val = np.clip(c_val, 1e-12, m_val * (1 - 1e-10))

                    # Compute RHS: beta * R * sum_z' P[z,z'] * u'(c'(m', z'))
                    rhs_sum = 0.0
                    for zp in range(Z):
                        m_prime = R * (m_val - c_val) + income[zp]
                        m_prime = np.clip(m_prime, 0.001, 40.0)
                        log_m_prime = np.log(m_prime)
                        # Get log(c_next) from spline
                        log_c_next = next_splines[zp](log_m_prime)
                        log_c_next = np.clip(log_c_next, -27.0, np.log(40.0))  # clip to valid range
                        c_next = np.exp(log_c_next)
                        c_next = np.clip(c_next, 1e-12, 40.0)
                        rhs_sum += transition[z, zp] * utility_prime(c_next)

                    rhs = beta * R * rhs_sum
                    if rhs <= 0:
                        log_c_new[i] = np.log(1e-12)
                    elif abs(gamma - 1.0) < 1e-10:
                        log_c_new[i] = -np.log(rhs)
                    else:
                        log_c_new[i] = -np.log(rhs) / gamma

                    # Clip to valid range
                    log_c_new[i] = np.clip(log_c_new[i], np.log(1e-12), np.log(m_val * (1 - 1e-10)))

                # Check convergence
                max_diff = np.max(np.abs(log_c_new - log_c))
                log_c = log_c_new
                if max_diff < 1e-10:
                    break

            c_policy_log[t, z, :] = log_c

            # Compute V[t][z]
            for i in range(n_fine):
                m_val = m_fine[i]
                c_val = np.exp(log_c[i])
                c_val = np.clip(c_val, 1e-12, m_val * (1 - 1e-10))

                cont_val = 0.0
                for zp in range(Z):
                    m_prime = R * (m_val - c_val) + income[zp]
                    m_prime = np.clip(m_prime, 0.001, 40.0)
                    log_m_prime = np.log(m_prime)
                    # Interpolate V from next period
                    V_next_zp = np.interp(log_m_prime, log_m_fine, V[t + 1, zp, :])
                    cont_val += transition[z, zp] * V_next_zp

                V[t, z, i] = utility(c_val) + beta * cont_val

        # Check time
        if time.time() > deadline - 5:
            # Fill remaining with proportional consumption
            for tt in range(t - 1, -1, -1):
                for z in range(Z):
                    c_policy_log[tt, z, :] = log_m_fine - np.log(2)
                    V[tt, z, :] = V[tt + 1, z, :]
            break

    # Extract consumption policy on the original resource_grid
    consumption = np.zeros((horizon, Z, M))

    for t in range(horizon):
        for z in range(Z):
            # Interpolate log_c from log_m_fine to log(m_grid)
            log_m_grid = np.log(m_grid)
            log_c_grid = np.interp(log_m_grid, log_m_fine, c_policy_log[t, z, :])
            c_grid = np.exp(log_c_grid)
            consumption[t, z, :] = c_grid

    # Terminal period: consume all
    for z in range(Z):
        consumption[horizon - 1, z, :] = m_grid.copy()

    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid + 1e-10)

    # Ensure finite values
    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=m_grid, neginf=1e-12)

    return consumption
