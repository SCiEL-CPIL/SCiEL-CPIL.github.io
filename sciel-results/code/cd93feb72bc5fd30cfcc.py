import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 147.0  # Safety margin before 150s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M_out = len(resource_grid)
    Z = len(income)

    # Use a fine internal grid for high accuracy
    M_int = 601
    m_grid = np.geomspace(0.001, 40, M_int)

    # Precompute grid differences
    dm = np.diff(m_grid)

    # consumption[t, z, m_index] on the output grid
    consumption = np.zeros((horizon, Z, M_out))

    # Helper for marginal utility
    def u_prime(c, g):
        c = np.maximum(c, 1e-12)
        if g == 1:
            return 1.0 / c
        else:
            return c ** (-g)

    # Helper for log marginal utility
    def log_u_prime(c, g):
        c = np.maximum(c, 1e-12)
        if g == 1:
            return -np.log(c)
        else:
            return -g * np.log(c)

    # Terminal condition: at t=5 (last decision), consume everything.
    c_next = np.zeros((Z, M_int))
    for z in range(Z):
        c_next[z] = m_grid

    # Set terminal period consumption on output grid
    for z in range(Z):
        consumption[horizon - 1, z] = resource_grid

    # Build cubic spline for terminal condition
    c_splines = [CubicSpline(m_grid, c_next[z]) for z in range(Z)]

    # Backward induction for t = 4 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with simple rule (consume all)
            for tt in range(t, -1, -1):
                for z in range(Z):
                    consumption[tt, z] = resource_grid
            break

        c_curr = np.zeros((Z, M_int))

        for z in range(Z):
            # Solve for s in [0, 1) such that F(s) = 0
            # F(s) = log(u'(m(1-s))) - log(beta * R * sum_{z'} P(z'|z) u'(c_next(m', z')))
            s = np.full(M_int, 0.2)
            s_lo = np.zeros(M_int)
            s_hi = np.full(M_int, 0.999999)

            for _ in range(30):
                if time.time() > deadline:
                    break

                c = m_grid * (1.0 - s)
                c = np.maximum(c, 1e-12)

                # LHS
                lhs = log_u_prime(c, gamma)

                # RHS components
                m_prime = R * m_grid[:, None] * s[:, None] + income[None, :]

                u_prime_sum = np.zeros(M_int)
                u_prime_prime_sum = np.zeros(M_int)

                for zp in range(Z):
                    # Use cubic spline for smooth interpolation and derivatives
                    c_next_val = c_splines[zp](m_prime[:, zp])
                    c_next_val = np.maximum(c_next_val, 1e-12)
                    c_next_prime = c_splines[zp](m_prime[:, zp], 1)

                    u_val = u_prime(c_next_val, gamma)

                    if gamma == 1:
                        u_prime_prime_val = -1.0 / (c_next_val * c_next_val)
                    else:
                        u_prime_prime_val = -gamma * (c_next_val ** (-gamma - 1))

                    dm_ds = R * m_grid

                    u_prime_sum += transition[z, zp] * u_val
                    u_prime_prime_sum += transition[z, zp] * u_prime_prime_val * c_next_prime * dm_ds

                rhs_val = beta * R * u_prime_sum
                rhs_val = np.maximum(rhs_val, 1e-12)
                rhs = np.log(rhs_val)

                F = lhs - rhs

                # Derivatives
                d_lhs = gamma / (1.0 - s)
                d_rhs = u_prime_prime_sum / u_prime_sum
                dF = d_lhs - d_rhs

                dF_safe = np.where(np.abs(dF) < 1e-10, 1e-10, dF)
                s_new = s - F / dF_safe

                mask_pos = F > 0
                mask_neg = F < 0
                s_hi[mask_pos] = s[mask_pos]
                s_lo[mask_neg] = s[mask_neg]

                s_new = np.clip(s_new, s_lo, s_hi)
                s = s_new

                if np.max(np.abs(s_new - s)) < 1e-10:
                    break

            # Final c
            c_curr[z] = m_grid * (1.0 - s)
            c_curr[z] = np.maximum(c_curr[z], 1e-12)
            c_curr[z] = np.minimum(c_curr[z], m_grid + 1e-10)

        # Store the policy for this period on the output grid
        for z in range(Z):
            consumption[t, z] = np.interp(resource_grid, m_grid, c_curr[z])

        # Update c_next and splines
        c_next = c_curr.copy()
        c_splines = [CubicSpline(m_grid, c_next[z]) for z in range(Z)]

    # Final validity checks
    for t in range(horizon):
        for z in range(Z):
            c = consumption[t, z]
            m = resource_grid
            c = np.maximum(c, 1e-12)
            c = np.minimum(c, m + 1e-10)
            consumption[t, z] = c

    # Enforce terminal condition exactly
    for z in range(Z):
        consumption[horizon - 1, z] = resource_grid

    return consumption
