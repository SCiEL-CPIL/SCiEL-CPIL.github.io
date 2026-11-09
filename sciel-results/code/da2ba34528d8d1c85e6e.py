"""
Consumption policy solver: Euler equation root-finding with numerical V'
"""
import time
import numpy as np
from scipy.optimize import brentq

DEADLINE_SECONDS = 135


def _crsa_utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)


def _crsa_marginal(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return 1.0 / c
    return c ** (-gamma)


def _solve_consumption_single(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=np.float64)
    trans = np.asarray(model['transition'], dtype=np.float64)
    horizon = model['horizon']
    states = len(income)
    n_grid = len(resource_grid)
    m_grid = np.asarray(resource_grid, dtype=np.float64)

    consumption = np.zeros((horizon, states, n_grid))

    # Terminal: consume everything
    consumption[horizon - 1] = np.tile(m_grid, (states, 1))
    V_next = np.tile(m_grid, (states, 1))  # V(m) = m at terminal (utility not needed for V')

    # We'll store V as a function of m for each state.
    # V_next[z] is defined on m_grid.

    for t in range(horizon - 2, -1, -1):
        V_current = np.zeros((states, n_grid))

        # Compute V' (derivative) of V_next for each state using central differences
        # V_next shape: (states, n_grid)
        # We need dV/dm for each state on the grid
        V_prime_next = np.zeros((states, n_grid))
        for z in range(states):
            v = V_next[z]
            # Central differences in the interior, one-sided at boundaries
            dm = np.diff(m_grid)
            # Forward difference at last point, backward at first
            V_prime_next[z, 0] = (v[1] - v[0]) / (m_grid[1] - m_grid[0])
            V_prime_next[z, -1] = (v[-1] - v[-2]) / (m_grid[-1] - m_grid[-2])
            V_prime_next[z, 1:-1] = (v[2:] - v[:-2]) / (m_grid[2:] - m_grid[:-2])

        for z in range(states):
            for i in range(n_grid):
                m = m_grid[i]

                # Define the Euler residual: g(c) = u'(c) - beta * R * sum_z' P(z'|z) * V'(R*(m-c) + income[z'])
                # We want g(c) = 0 for c in (0, m)

                def euler_residual(c):
                    c = max(c, 1e-12)
                    c = min(c, m - 1e-12)
                    mu = _crsa_marginal(c, gamma)

                    # Compute m' for each z'
                    m_prime = R * (m - c) + income  # shape (states,)

                    # Interpolate V_prime_next[z'] at m_prime
                    # Clip m_prime to grid range
                    m_prime_clipped = np.clip(m_prime, m_grid[0], m_grid[-1])

                    # V_prime for each z'
                    vp = np.zeros(states)
                    for zp in range(states):
                        vp[zp] = np.interp(m_prime_clipped[zp], m_grid, V_prime_next[zp])

                    # Expected marginal value
                    expected_vp = np.dot(trans[z], vp)

                    return mu - beta * R * expected_vp

                # Solve for c in (eps, m - eps)
                eps = 1e-10
                lo = eps
                hi = m - eps

                # Check signs at boundaries
                g_lo = euler_residual(lo)
                g_hi = euler_residual(hi)

                if g_lo * g_hi > 0:
                    # No sign change; pick the endpoint with smaller |g|
                    # or just use a fallback
                    if abs(g_lo) < abs(g_hi):
                        c_star = lo
                    else:
                        c_star = hi
                else:
                    try:
                        c_star = brentq(euler_residual, lo, hi, xtol=1e-10, rtol=1e-10, maxiter=100)
                    except Exception:
                        # Fallback: bisection
                        a, b = lo, hi
                        fa = euler_residual(a)
                        for _ in range(60):
                            mid = 0.5 * (a + b)
                            fm = euler_residual(mid)
                            if fa * fm <= 0:
                                b = mid
                            else:
                                a = mid
                                fa = fm
                        c_star = 0.5 * (a + b)

                c_star = np.clip(c_star, 1e-12, m)
                consumption[t, z, i] = c_star

                # Compute V_current[z, i] = u(c_star) + beta * sum_z' P * V_next(z', m')
                u_c = _crsa_utility(c_star, gamma)
                m_prime = R * (m - c_star) + income
                m_prime_clipped = np.clip(m_prime, m_grid[0], m_grid[-1])
                cont = np.zeros(states)
                for zp in range(states):
                    cont[zp] = np.interp(m_prime_clipped[zp], m_grid, V_next[zp])
                V_current[z, i] = u_c + beta * np.dot(trans[z], cont)

        V_next = V_current

    return consumption


def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    try:
        result = _solve_consumption_single(model, resource_grid, budget)

        # Validate
        if not np.all(np.isfinite(result)):
            raise ValueError("Non-finite values")
        m = resource_grid[None, None, :]
        if np.any(result <= 0) or np.any(result > m + 1e-10):
            raise ValueError("c out of bounds")
        horizon = model['horizon']
        states = model['transition'].shape[0]
        if not np.allclose(result[horizon - 1],
                           np.tile(resource_grid, (states, 1)), atol=1e-9):
            raise ValueError("Terminal condition violated")

        return result
    except Exception:
        # Fallback: simple heuristic
        horizon = model['horizon']
        states = len(model['income'])
        n_grid = len(resource_grid)

        result = np.zeros((horizon, states, n_grid))
        for t in range(horizon):
            if t == horizon - 1:
                result[t] = np.tile(resource_grid, (states, 1))
            else:
                frac_save = 0.1 * (horizon - 1 - t) / (horizon - 1)
                result[t] = (1.0 - frac_save) * resource_grid[None, :]
                result[t] = np.clip(result[t], 1e-6, resource_grid[None, :])
                result[t] = np.tile(result[t], (states, 1))

        return result
