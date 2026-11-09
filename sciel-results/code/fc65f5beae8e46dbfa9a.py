"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq

def _utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return np.log(c)
    return c ** (1 - gamma) / (1 - gamma)

def _utility_prime(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return 1.0 / c
    return c ** (-gamma)

def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    log_m = np.log(resource_grid)
    
    # Terminal condition: consume all at t=horizon-1 (index 5)
    # consumption[t, z, i] for t in 0..horizon-1
    consumption = np.zeros((horizon, n_states, n_grid))
    consumption[horizon - 1] = resource_grid[None, None, :]
    
    # Value function: V[t, z, i] = max utility from t onwards given m, z
    # We'll store value and its derivative for FOC solving
    V = np.zeros((horizon, n_states, n_grid))
    V[horizon - 1] = _utility(resource_grid, gamma)[None, None, :]
    Vp = np.zeros((horizon, n_states, n_grid))
    Vp[horizon - 1] = _utility_prime(resource_grid, gamma)[None, None, :]
    
    # For each period from horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break
        for z in range(n_states):
            # Compute continuation value: E[V[t+1, z', m']]
            # m' = R*(m - c) + income[z']
            # V[t, z, m] = max_c u(c) + beta * E_z'[V[t+1, z', R*(m-c) + income[z']]]
            
            # For each grid point m_i, find optimal c_i
            # FOC: u'(c) = beta * sum_{z'} P(z'|z) * V'[t+1, z', R*(m-c)+income[z']]
            
            # We need to interpolate V and Vp for the next period
            # Use log-space interpolation for smoothness
            
            c_opt = np.zeros(n_grid)
            V[t, z] = np.zeros(n_grid)
            Vp[t, z] = np.zeros(n_grid)
            
            for i in range(n_grid):
                m_i = resource_grid[i]
                if m_i < 1e-10:
                    c_opt[i] = 1e-12
                    V[t, z, i] = _utility(1e-12, gamma)
                    Vp[t, z, i] = _utility_prime(1e-12, gamma)
                    continue
                
                # Define FOC function: f(c) = u'(c) - beta * sum_z' P(z'|z) * Vp_next(z', R*(m-c)+income[z'])
                # c in (0, m)
                
                def foc_residual(c):
                    c_safe = max(c, 1e-12)
                    lhs = _utility_prime(c_safe, gamma)
                    rhs = 0.0
                    for zp in range(n_states):
                        m_prime = R * (m_i - c_safe) + income[zp]
                        if m_prime < 1e-10:
                            m_prime = 1e-10
                        # Interpolate Vp[t+1, zp, :] in log-space
                        log_mp = np.log(m_prime)
                        if log_mp <= log_m[0]:
                            vp_val = Vp[t+1, zp, 0]
                        elif log_mp >= log_m[-1]:
                            vp_val = Vp[t+1, zp, -1]
                        else:
                            vp_val = np.interp(log_mp, log_m, Vp[t+1, zp, :])
                        rhs += transition[z, zp] * vp_val
                    return lhs - beta * rhs
                
                # Find c in (eps, m_i - eps)
                eps = 1e-10
                c_lo = eps
                c_hi = m_i - eps
                
                try:
                    f_lo = foc_residual(c_lo)
                    f_hi = foc_residual(c_hi)
                    
                    if f_lo < 0:  # u'(c) < beta*E[V'] at low c -> increase c
                        c_opt[i] = c_lo
                    elif f_hi > 0:  # u'(c) > beta*E[V'] at high c -> decrease c
                        c_opt[i] = c_hi
                    else:
                        c_opt[i] = brentq(foc_residual, c_lo, c_hi, xtol=1e-12, rtol=1e-10)
                except:
                    # Fallback: bisection
                    a, b = c_lo, c_hi
                    for _ in range(50):
                        mid = (a + b) / 2
                        f_mid = foc_residual(mid)
                        if f_mid < 0:
                            a = mid
                        else:
                            b = mid
                    c_opt[i] = (a + b) / 2
                
                c_opt[i] = np.clip(c_opt[i], 1e-12, m_i)
                
                # Compute value
                V[t, z, i] = _utility(c_opt[i], gamma)
                # Envelope theorem: V'(m) = u'(c) = beta * E[V']
                Vp[t, z, i] = _utility_prime(c_opt[i], gamma)
            
            consumption[t, z] = c_opt
    
    # Final constraints
    consumption = np.clip(consumption, 1e-12, None)
    for t in range(horizon):
        for z in range(n_states):
            consumption[t, z] = np.minimum(consumption[t, z], resource_grid + 1e-10)
    
    # Ensure terminal condition
    consumption[horizon - 1] = resource_grid[None, None, :]
    
    # Ensure finite
    consumption = np.where(np.isfinite(consumption), consumption, 1e-12)
    
    return consumption
# EVOLVE-BLOCK-END
