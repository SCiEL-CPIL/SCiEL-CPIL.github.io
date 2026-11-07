import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 135.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = len(resource_grid)
    Z = len(income)
    m_grid = resource_grid

    # CRRA utility and derivatives
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            c = np.maximum(c, 1e-12)
            return np.log(c)
        def utility_prime(c):
            c = np.maximum(c, 1e-12)
            return 1.0 / c
        def utility_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -1.0 / (c * c)
    else:
        def utility(c):
            c = np.maximum(c, 1e-12)
            return c**(1 - gamma) / (1 - gamma)
        def utility_prime(c):
            c = np.maximum(c, 1e-12)
            return c**(-gamma)
        def utility_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -gamma * c**(-gamma - 1)

    # Use a fine grid for computation
    m_fine = np.geomspace(0.001, 40, 801)
    n_fine = len(m_fine)

    # V[t][z] will be a 1D array of length n_fine
    V = np.zeros((horizon, Z, n_fine))
    c_policy = np.zeros((horizon, Z, n_fine))

    # Terminal: consume all
    for z in range(Z):
        V[horizon - 1, z, :] = utility(m_fine)
        c_policy[horizon - 1, z, :] = m_fine.copy()

    # Build spline interpolators for the terminal period
    c_splines = [None] * Z
    V_splines = [None] * Z
    for z in range(Z):
        c_splines[z] = CubicSpline(m_fine, c_policy[horizon - 1, z, :], bc_type='natural')
        V_splines[z] = CubicSpline(m_fine, V[horizon - 1, z, :], bc_type='natural')

    # Work backwards from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        for z in range(Z):
            c_t_z = np.zeros(n_fine)
            V_t_z = np.zeros(n_fine)

            for i in range(n_fine):
                m_val = m_fine[i]

                # Initial guess: proportional consumption
                c = max(0.5 * m_val, 1e-8)
                c = min(c, m_val * 0.999)

                # Secant method on the Euler equation
                c_prev = c * 0.95
                F_prev = 0.0
                for zp in range(Z):
                    m_prime_prev = R * (m_val - c_prev) + income[zp]
                    m_prime_prev = np.clip(m_prime_prev, m_fine[0], m_fine[-1])
                    c_next_prev = float(c_splines[zp](m_prime_prev))
                    c_next_prev = max(c_next_prev, 1e-12)
                    F_prev += transition[z, zp] * utility_prime(c_next_prev)
                F_prev = utility_prime(c_prev) - beta * R * F_prev

                for newton_iter in range(20):
                    # Compute F(c)
                    F = 0.0
                    for zp in range(Z):
                        m_prime = R * (m_val - c) + income[zp]
                        m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                        c_next = float(c_splines[zp](m_prime))
                        c_next = max(c_next, 1e-12)
                        F += transition[z, zp] * utility_prime(c_next)
                    F = utility_prime(c) - beta * R * F

                    # Secant step
                    if abs(F - F_prev) > 1e-15:
                        c_new = c - F * (c - c_prev) / (F - F_prev)
                    else:
                        c_new = c

                    c_new = np.clip(c_new, 1e-12, m_val * (1 - 1e-10))

                    c_prev = c
                    F_prev = F
                    c = c_new

                    if abs(F) < 1e-10:
                        break

                c_t_z[i] = c

                # Compute V_t_z[i]
                cont_val = 0.0
                for zp in range(Z):
                    m_prime = R * (m_val - c) + income[zp]
                    m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                    cont_val += transition[z, zp] * float(V_splines[zp](m_prime))
                V_t_z[i] = utility(c) + beta * cont_val

            c_policy[t, z, :] = c_t_z
            V[t, z, :] = V_t_z

            # Check time
            if time.time() > deadline - 10:
                break

        # Build new splines for the next iteration
        for z in range(Z):
            c_splines[z] = CubicSpline(m_fine, c_policy[t, z, :], bc_type='natural')
            V_splines[z] = CubicSpline(m_fine, V[t, z, :], bc_type='natural')

    # Extract consumption policy on the original resource_grid using cubic splines
    consumption = np.zeros((horizon, Z, M))

    # Terminal period
    for z in range(Z):
        consumption[horizon - 1, z, :] = m_grid.copy()

    # For t = 0 to horizon-2, interpolate from the fine grid policy using cubic splines
    for t in range(horizon - 1):
        for z in range(Z):
            try:
                cs = CubicSpline(m_fine, c_policy[t, z, :], bc_type='natural')
                consumption[t, z, :] = cs(m_grid)
            except Exception:
                consumption[t, z, :] = np.interp(m_grid, m_fine, c_policy[t, z, :])

    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid + 1e-10)

    # Ensure finite values
    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=m_grid, neginf=1e-12)

    return consumption
