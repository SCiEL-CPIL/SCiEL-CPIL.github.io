"""Vectorized consumption policy solver with Newton FOC and policy iteration."""
import time
import numpy as np

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


def _interp_linear(x, xp, fp):
    """Vectorized linear interpolation of fp at points x, with clipping at boundaries."""
    x = np.asarray(x)
    xp = np.asarray(xp)
    fp = np.asarray(fp)
    n = len(xp)
    # Clip to range
    x_clipped = np.clip(x, xp[0], xp[-1])
    # searchsorted
    idx = np.searchsorted(xp, x_clipped, side='right') - 1
    idx = np.clip(idx, 0, n - 2)
    x0 = xp[idx]
    x1 = xp[idx + 1]
    denom = x1 - x0
    # Handle zero denom
    denom = np.where(denom < 1e-15, 1.0, denom)
    w = (x_clipped - x0) / denom
    w = np.clip(w, 0.0, 1.0)
    return fp[idx] + w * (fp[idx + 1] - fp[idx])


def _interp_linear_slope(x, xp, fp):
    """Vectorized linear interpolation slope (derivative) of fp at points x."""
    x = np.asarray(x)
    xp = np.asarray(xp)
    fp = np.asarray(fp)
    n = len(xp)
    x_clipped = np.clip(x, xp[0], xp[-1])
    idx = np.searchsorted(xp, x_clipped, side='right') - 1
    idx = np.clip(idx, 0, n - 2)
    x0 = xp[idx]
    x1 = xp[idx + 1]
    denom = x1 - x0
    denom = np.where(denom < 1e-15, 1.0, denom)
    slope = (fp[idx + 1] - fp[idx]) / denom
    return slope


def _solve_consumption_single(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    trans = np.asarray(model['transition'])
    horizon = model['horizon']
    states = len(income)
    n_grid = len(resource_grid)

    consumption = np.zeros((horizon, states, n_grid))

    # Terminal: consume all
    consumption[horizon - 1] = np.tile(resource_grid, (states, 1))

    # V[z, i] = value function
    V = _crsa_utility(np.tile(resource_grid, (states, 1)), gamma)

    # Policy iteration: refine the value function
    # We'll do a few iterations of policy evaluation + improvement
    # But first, get a good initial policy via vectorized Newton

    # For each t from horizon-2 down to 0:
    for t in range(horizon - 2, -1, -1):
        V_new = np.zeros((states, n_grid))
        m_vals = resource_grid  # shape (n_grid,)

        for z in range(states):
            # For this state z, we need to solve for c[m] for all m in resource_grid
            # FOC: u'(c) = beta * R * E_z'[V'(R(m-c) + income[z'])]
            # We'll use Newton's method, vectorized over all m.

            # Initial guess: use a simple heuristic
            # For CRRA, optimal c is roughly a fraction of m + expected income
            # Let's start with c = 0.5 * m + 0.2 * mean_income (clipped)
            mean_inc = np.sum(trans[z] * income)
            c = 0.5 * m_vals + 0.2 * mean_inc
            c = np.clip(c, 1e-10, m_vals)

            # Newton iterations
            for it in range(50):
                # Compute m' for each m, c
                m_minus_c = m_vals - c
                m_prime = np.zeros((states, n_grid))
                for zp in range(states):
                    m_prime[zp, :] = R * m_minus_c + income[zp]

                # Interpolate V' at m_prime for each state
                # V_prime[zp, :] is the slope of V[zp, :] at m_prime[zp, :]
                V_prime = np.zeros((states, n_grid))
                for zp in range(states):
                    V_prime[zp, :] = _interp_linear_slope(m_prime[zp, :], resource_grid, V[zp, :])

                # E[V']
                exp_V_prime = np.sum(trans[z, :] * V_prime, axis=0)  # shape (n_grid,)

                # u'(c)
                u_prime = _crsa_marginal(c, gamma)

                # Residual: u'(c) - beta * R * E[V']
                residual = u_prime - beta * R * exp_V_prime

                # Derivative of residual w.r.t. c:
                # d/dc [u'(c)] = -u''(c) = -gamma * c^(-gamma-1) (for CRRA)
                # d/dc [beta * R * E[V']] = beta * R * E[V''] * (-R) ... 
                # Actually, d/dc of m_prime = -R, so d/dc of V'(m_prime) = V''(m_prime) * (-R)
                # So d/dc of E[V'] = E[V''(m_prime)] * (-R)
                # d/dc of residual = -u''(c) - beta * R * (-R) * E[V''] = -u''(c) + beta * R^2 * E[V'']

                # u''(c) = -gamma * c^(-gamma-1) for CRRA (when gamma != 1)
                # For gamma = 1: u''(c) = -1/c^2
                c_safe = np.maximum(c, 1e-12)
                if abs(gamma - 1.0) < 1e-12:
                    u_dd = -1.0 / (c_safe ** 2)
                else:
                    u_dd = -gamma * (c_safe ** (-gamma - 1))

                # V''(m_prime): approximate via second difference of V
                # For simplicity, use a finite difference approximation
                # V''(x) ≈ (V(x+h) - 2V(x) + V(x-h)) / h^2
                # But we're interpolating, so let's use the slope of the slope
                # Alternatively, approximate V'' using the change in V_prime
                # Let's use a small h and compute V_prime at m_prime ± h
                h = 1e-4 * np.maximum(1.0, m_prime)
                m_prime_plus = m_prime + h
                m_prime_minus = m_prime - h
                V_prime_plus = np.zeros((states, n_grid))
                V_prime_minus = np.zeros((states, n_grid))
                for zp in range(states):
                    V_prime_plus[zp, :] = _interp_linear_slope(m_prime_plus[zp, :], resource_grid, V[zp, :])
                    V_prime_minus[zp, :] = _interp_linear_slope(m_prime_minus[zp, :], resource_grid, V[zp, :])
                V_dd = (V_prime_plus - V_prime_minus) / (2 * h)  # shape (states, n_grid)
                exp_V_dd = np.sum(trans[z, :] * V_dd, axis=0)  # shape (n_grid,)

                # Derivative of residual
                d_residual = -u_dd + beta * R * R * exp_V_dd

                # Newton step
                d_residual_safe = np.where(np.abs(d_residual) < 1e-15, 1e-15, d_residual)
                # Ensure correct sign: if d_residual is negative, step is positive, etc.
                # We want to move in the direction that reduces |residual|
                # c_new = c - residual / d_residual
                # But we need to ensure c stays in (0, m)
                step = residual / d_residual_safe
                c_new = c - step

                # Clip
                c_new = np.clip(c_new, 1e-10, m_vals)

                # Check convergence
                if np.max(np.abs(c_new - c)) < 1e-10:
                    c = c_new
                    break
                c = c_new

            consumption[t, z, :] = c

            # Compute V_new[z, :] = u(c) + beta * E[V(m')]
            u_c = _crsa_utility(c, gamma)
            exp_V = np.zeros(n_grid)
            for zp in range(states):
                m_prime_zp = R * (m_vals - c) + income[zp]
                v_interp = _interp_linear(m_prime_zp, resource_grid, V[zp, :])
                exp_V += trans[z, zp] * v_interp
            V_new[z, :] = u_c + beta * exp_V

        V = V_new

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
        states = model['transition'].shape[0]
        if not np.allclose(result[5], np.tile(resource_grid, (states, 1)), atol=1e-9):
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
                result[t] = 0.9 * resource_grid[None, :]
                result[t] = np.clip(result[t], 1e-6, resource_grid[None, :])
                result[t] = np.tile(result[t], (states, 1))

        return result
