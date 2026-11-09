"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    grid = np.asarray(resource_grid, dtype=float)
    n_grid = len(grid)
    n_z = len(income)

    is_log = abs(gamma - 1.0) < 1e-10

    def u(c):
        c = np.maximum(c, 1e-12)
        if is_log:
            return np.log(c)
        return c ** (1.0 - gamma) / (1.0 - gamma)

    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if is_log:
            return 1.0 / c
        return c ** (-gamma)

    # Value and policy arrays
    V = np.zeros((horizon, n_z, n_grid))
    C = np.zeros((horizon, n_z, n_grid))

    # Terminal period: consume everything
    C[horizon - 1, :, :] = grid[np.newaxis, np.newaxis, :]
    V[horizon - 1, :, :] = u(grid)[np.newaxis, np.newaxis, :]

    # We'll use a hybrid approach:
    # 1. Coarse grid search to get a good initial bracket
    # 2. Brent's method for precise root finding
    # 3. If FOC fails, fall back to dense grid search

    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fill remaining with simple heuristic
            for tt in range(t, -1, -1):
                for z in range(n_z):
                    remaining = tt + 1
                    # Simple rule: consume fraction based on remaining horizon
                    frac = 1.0 / (remaining * beta ** (remaining - 1))
                    frac = min(max(frac, 0.001), 1.0)
                    C[tt, z, :] = grid * frac
                    V[tt, z, :] = V[tt + 1, z, :] if tt + 1 < horizon else u(grid)
            break

        # Get next period's value function
        V_next = V[t + 1]  # (n_z, n_grid)
        C_next = C[t + 1]  # (n_z, n_grid)
        
        # By envelope theorem, Vp[t+1] = u'(C[t+1])
        Vp_next = u_prime(np.maximum(C_next, 1e-12))  # (n_z, n_grid)

        for z in range(n_z):
            c_opt = np.zeros(n_grid)
            
            for i in range(n_grid):
                m = grid[i]
                if m < 1e-10:
                    c_opt[i] = 1e-12
                    continue

                # Define FOC residual function
                def foc_residual(c, m_val=m, z_val=z):
                    c_safe = max(c, 1e-12)
                    lhs = u_prime(c_safe)
                    rhs = 0.0
                    for zp in range(n_z):
                        m_prime = R * (m_val - c_safe) + income[zp]
                        if m_prime < grid[0]:
                            m_prime = grid[0]
                        elif m_prime > grid[-1]:
                            m_prime = grid[-1]
                        vp = np.interp(m_prime, grid, Vp_next[zp, :])
                        rhs += transition[z_val, zp] * vp
                    return lhs - beta * R * rhs

                # Coarse scan to find sign change
                n_coarse = 50
                ratios = np.linspace(0.001, 1.0, n_coarse)
                
                f_prev = None
                c_prev = None
                c_lo = 1e-12
                c_hi = m
                found = False
                
                for ratio in ratios:
                    c = min(max(ratio * m, 1e-12), m)
                    f_val = foc_residual(c)
                    if f_prev is not None:
                        if f_prev * f_val <= 0:
                            # Sign change found
                            a, b = c_prev, c
                            try:
                                c_opt[i] = brentq(foc_residual, a, b, xtol=1e-12, rtol=1e-10, maxiter=100)
                            except Exception:
                                # Bisection fallback
                                lo, hi = a, b
                                for _ in range(60):
                                    mid = 0.5 * (lo + hi)
                                    fm = foc_residual(mid)
                                    if fm < 0:
                                        lo = mid
                                    else:
                                        hi = mid
                                c_opt[i] = 0.5 * (lo + hi)
                            found = True
                            break
                    f_prev = f_val
                    c_prev = c

                if not found:
                    # No sign change: check if FOC is always positive or negative
                    f_lo = foc_residual(c_lo)
                    f_hi = foc_residual(c_hi)
                    
                    if f_lo < 0:
                        # FOC negative at low c, so optimum is at lower bound
                        c_opt[i] = c_lo
                    elif f_hi > 0:
                        # FOC positive at high c, so optimum is at upper bound
                        c_opt[i] = c_hi
                    else:
                        # Fallback: dense grid search
                        n_fine = 300
                        fine_ratios = np.linspace(0.001, 1.0, n_fine)
                        best_c = c_lo
                        best_val = -np.inf
                        for ratio in fine_ratios:
                            c = min(max(ratio * m, 1e-12), m)
                            cont = 0.0
                            for zp in range(n_z):
                                m_prime = R * (m - c) + income[zp]
                                if m_prime < grid[0]:
                                    m_prime = grid[0]
                                elif m_prime > grid[-1]:
                                    m_prime = grid[-1]
                                vv = np.interp(m_prime, grid, V_next[zp, :])
                                cont += transition[z, zp] * vv
                            total = u(c) + beta * cont
                            if total > best_val:
                                best_val = total
                                best_c = c
                        c_opt[i] = best_c

                c_opt[i] = np.clip(c_opt[i], 1e-12, m)

            C[t, z, :] = c_opt
            
            # Compute value function
            for i in range(n_grid):
                m = grid[i]
                c = c_opt[i]
                cont = 0.0
                for zp in range(n_z):
                    m_prime = R * (m - c) + income[zp]
                    if m_prime < grid[0]:
                        m_prime = grid[0]
                    elif m_prime > grid[-1]:
                        m_prime = grid[-1]
                    vv = np.interp(m_prime, grid, V_next[zp, :])
                    cont += transition[z, zp] * vv
                V[t, z, i] = u(c) + beta * cont

    return _finalize(C, grid, horizon, n_z, n_grid)

def _finalize(C, grid, horizon, n_z, n_grid):
    # Ensure terminal condition
    C[horizon - 1, :, :] = grid[np.newaxis, np.newaxis, :]
    # Ensure 0 < c <= m + 1e-10 and finite
    for t in range(horizon):
        for z in range(n_z):
            C[t, z, :] = np.clip(C[t, z, :], 1e-12, grid + 1e-10)
    C = np.where(np.isfinite(C), C, 1e-12)
    return C
# EVOLVE-BLOCK-END
