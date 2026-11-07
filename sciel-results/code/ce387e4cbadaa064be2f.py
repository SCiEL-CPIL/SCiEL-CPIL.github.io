"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    m_grid = resource_grid  # shape (101,)
    n_m = len(m_grid)
    n_z = len(income)
    
    # Define utility and marginal utility functions
    if abs(gamma - 1.0) < 1e-10:
        def util(c):
            c = np.maximum(c, 1e-12)
            return np.log(c)
        def mutil(c):
            c = np.maximum(c, 1e-12)
            return 1.0 / c
    else:
        def util(c):
            c = np.maximum(c, 1e-12)
            return (c ** (1 - gamma) - 1) / (1 - gamma)
        def mutil(c):
            c = np.maximum(c, 1e-12)
            return c ** (-gamma)
    
    # Allocate storage
    c_policy = np.zeros((horizon, n_z, n_m))
    
    # Terminal condition: at t = horizon-1, consume everything
    c_policy[horizon-1, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    # V_next[z, m] = value function at t+1
    # At terminal (t = horizon-1), V = util(m)
    V_next = np.tile(util(m_grid), (n_z, 1))  # shape (n_z, n_m)
    
    # Backward induction from t = horizon-2 down to 0
    for t in range(horizon-2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: use simple rule for remaining periods
            for tt in range(t, -1, -1):
                for z in range(n_z):
                    c_policy[tt, z, :] = m_grid * 0.5
            break
        
        V_curr = np.zeros((n_z, n_m))
        
        for z in range(n_z):
            if time.time() > deadline - 5:
                # Fallback
                for z2 in range(z, n_z):
                    c_policy[t, z2, :] = m_grid * 0.5
                    V_curr[z2, :] = util(m_grid)
                break
            
            # For this income state z, solve for optimal c at each m
            # FOC: mutil(c) = beta * R * sum_z' transition[z, z'] * mutil_next(z', m_next)
            # where m_next = R*(m-c) + income[z']
            
            # We'll use bisection/root-finding for each m
            # The function f(c) = mutil(c) - beta * R * sum_z' transition[z,z'] * mutil(z', R*(m-c)+income[z'])
            # is decreasing in c (mutil decreases, and the continuation term increases as c decreases)
            # So f(c) is strictly decreasing, giving a unique root.
            
            c_opt = np.zeros(n_m)
            
            for i in range(n_m):
                m = m_grid[i]
                
                # Define the FOC residual function
                def foc_residual(c, m=m, z=z):
                    # c is scalar
                    c_safe = max(c, 1e-12)
                    left = mutil(c_safe)
                    
                    # Compute expected marginal utility of next period
                    right = 0.0
                    for zp in range(n_z):
                        if transition[z, zp] < 1e-15:
                            continue
                        m_next = R * (m - c_safe) + income[zp]
                        m_next_safe = max(m_next, 1e-12)
                        right += transition[z, zp] * mutil(m_next_safe)
                    
                    return left - beta * R * right
                
                # Bisection method to find root
                # c is in (0, m)
                lo = 1e-12
                hi = m - 1e-12
                
                # Check if root is at boundary
                f_lo = foc_residual(lo)
                f_hi = foc_residual(hi)
                
                if f_lo < 0:
                    # At c=lo, left < right, so optimal c is very small
                    c_opt[i] = lo
                    continue
                if f_hi > 0:
                    # At c=hi, left > right, so optimal c is very large (consume almost all)
                    c_opt[i] = hi
                    continue
                
                # Bisection
                for _ in range(50):
                    mid = 0.5 * (lo + hi)
                    f_mid = foc_residual(mid)
                    if abs(f_mid) < 1e-14:
                        break
                    if f_mid > 0:
                        lo = mid
                    else:
                        hi = mid
                
                c_opt[i] = 0.5 * (lo + hi)
            
            # Ensure validity
            c_opt = np.clip(c_opt, 1e-12, m_grid + 1e-10)
            c_policy[t, z, :] = c_opt
            
            # Compute V_curr[z, m] = util(c) + beta * sum_z' transition[z,z'] * V_next[z', R*(m-c)+income[z']]
            for i in range(n_m):
                m = m_grid[i]
                c = c_opt[i]
                
                val = util(c)
                for zp in range(n_z):
                    if transition[z, zp] < 1e-15:
                        continue
                    m_next = R * (m - c) + income[zp]
                    # Interpolate V_next[zp] at m_next
                    if m_next <= m_grid[0]:
                        v_next = V_next[zp, 0]
                    elif m_next >= m_grid[-1]:
                        v_next = V_next[zp, -1]
                    else:
                        v_next = np.interp(m_next, m_grid, V_next[zp])
                    val += beta * transition[z, zp] * v_next
                
                V_curr[z, i] = val
        
        V_next = V_curr
    
    # Ensure c[5] = m_grid exactly
    c_policy[horizon-1, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    # Final validation
    if not np.all(np.isfinite(c_policy)):
        for t in range(horizon):
            for z in range(n_z):
                if t == horizon - 1:
                    c_policy[t, z, :] = m_grid
                else:
                    c_policy[t, z, :] = m_grid * 0.5
    
    # Ensure 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            c_policy[t, z, :] = np.clip(c_policy[t, z, :], 1e-12, m_grid + 1e-10)
    
    # Ensure c[5] = m_grid
    c_policy[horizon-1, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    return c_policy
# EVOLVE-BLOCK-END
