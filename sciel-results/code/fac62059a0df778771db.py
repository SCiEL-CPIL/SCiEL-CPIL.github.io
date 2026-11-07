"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    n_z = len(income)
    
    # Dense internal grid for backward induction
    m_grid = np.geomspace(0.001, 40, 251)
    n_m = len(m_grid)
    
    # Utility functions
    def u(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c_safe)
        else:
            return c_safe ** (1.0 - gamma) / (1.0 - gamma)
    
    def u_prime(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c_safe
        else:
            return c_safe ** (-gamma)
    
    # Policy array: c[t, z, m_index]
    c_policy = np.zeros((horizon, n_z, n_m))
    
    # Terminal: consume all
    for z in range(n_z):
        c_policy[horizon - 1, z, :] = m_grid
    
    # Value function on m_grid: V[t, z, m_index]
    V = np.zeros((horizon, n_z, n_m))
    for z in range(n_z):
        V[horizon - 1, z, :] = u(m_grid)
    
    # Precompute interpolation helpers for V[t+1]
    # Use np.interp with a slightly denser grid for smoother continuation values
    m_dense = np.geomspace(0.001, 40, 501)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10:
            break
        
        # Interpolate V[t+1, z] to dense grid for smooth continuation
        V_dense_next = np.zeros((n_z, len(m_dense)))
        for zp in range(n_z):
            V_dense_next[zp, :] = np.interp(m_dense, m_grid, V[t + 1, zp, :])
        
        # Precompute derivative of V_dense_next using central differences
        dV_dm = np.zeros((n_z, len(m_dense)))
        dm = np.diff(m_dense)
        for zp in range(n_z):
            # Central differences for interior points
            dV_dm[zp, 1:-1] = (V_dense_next[zp, 2:] - V_dense_next[zp, :-2]) / (m_dense[2:] - m_dense[:-2])
            # Forward difference for first point
            dV_dm[zp, 0] = (V_dense_next[zp, 1] - V_dense_next[zp, 0]) / dm[0]
            # Backward difference for last point
            dV_dm[zp, -1] = (V_dense_next[zp, -1] - V_dense_next[zp, -2]) / dm[-1]
        
        for z in range(n_z):
            for i in range(n_m):
                m_i = m_grid[i]
                
                if m_i < 1e-10:
                    c_policy[t, z, i] = m_i
                    V[t, z, i] = u(m_i)
                    continue
                
                def euler_residual(c):
                    if c <= 0 or c >= m_i - 1e-12:
                        return 1e10
                    m_save = R * (m_i - c)
                    total_v_prime = 0.0
                    for zp in range(n_z):
                        prob = transition[z, zp]
                        if prob < 1e-15:
                            continue
                        m_zp = m_save + income[zp]
                        m_zp = np.clip(m_zp, 1e-12, 40.0)
                        # Interpolate derivative from dense grid
                        v_prime_val = np.interp(m_zp, m_dense, dV_dm[zp, :])
                        total_v_prime += prob * v_prime_val
                    return u_prime(c) - beta * R * total_v_prime
                
                # Newton-bisection hybrid
                c_lo = 1e-10
                c_hi = m_i * (1 - 1e-10)
                if c_hi <= c_lo:
                    c_policy[t, z, i] = m_i
                    V[t, z, i] = u(m_i)
                    continue
                
                c_curr = m_i * 0.5
                for _ in range(30):
                    f_val = euler_residual(c_curr)
                    if abs(f_val) < 1e-12:
                        break
                    
                    h_newton = max(1e-8, c_curr * 1e-6)
                    f_plus = euler_residual(min(c_curr + h_newton, c_hi - 1e-12))
                    f_minus = euler_residual(max(c_curr - h_newton, c_lo + 1e-12))
                    df_val = (f_plus - f_minus) / (2 * h_newton)
                    
                    if abs(df_val) < 1e-15:
                        break
                    
                    c_new = c_curr - f_val / df_val
                    
                    if c_new <= c_lo or c_new >= c_hi:
                        if f_val > 0:
                            c_lo = c_curr
                        else:
                            c_hi = c_curr
                        c_new = (c_lo + c_hi) / 2
                    else:
                        if (c_new - c_curr) * f_val < 0:
                            if f_val > 0:
                                c_lo = c_curr
                            else:
                                c_hi = c_curr
                        else:
                            if f_val > 0:
                                c_hi = c_curr
                            else:
                                c_lo = c_curr
                    
                    c_curr = c_new
                    if c_hi - c_lo < 1e-12:
                        break
                
                c_opt = np.clip(c_curr, c_lo, c_hi)
                c_policy[t, z, i] = c_opt
                
                # Compute V[t, z, i]
                m_save = R * (m_i - c_opt)
                total = 0.0
                for zp in range(n_z):
                    prob = transition[z, zp]
                    if prob < 1e-15:
                        continue
                    m_zp = m_save + income[zp]
                    m_zp = np.clip(m_zp, 1e-12, 40.0)
                    v_val = np.interp(m_zp, m_dense, V_dense_next[zp, :])
                    total += prob * v_val
                V[t, z, i] = u(c_opt) + beta * total
    
    # Interpolate to resource_grid
    result = np.zeros((horizon, n_z, len(resource_grid)))
    for t in range(horizon):
        for z in range(n_z):
            result[t, z, :] = np.interp(resource_grid, m_grid, c_policy[t, z, :])
    
    # Enforce constraints
    for t in range(horizon):
        for z in range(n_z):
            result[t, z, :] = np.maximum(result[t, z, :], 1e-12)
            result[t, z, :] = np.minimum(result[t, z, :], resource_grid + 1e-10)
    
    # Terminal condition
    for z in range(n_z):
        result[horizon - 1, z, :] = resource_grid
    
    # Ensure finiteness
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=1e-12, posinf=1.0, neginf=1e-12)
        for t in range(horizon):
            for z in range(n_z):
                result[t, z, :] = np.maximum(result[t, z, :], 1e-12)
                result[t, z, :] = np.minimum(result[t, z, :], resource_grid + 1e-10)
        for z in range(n_z):
            result[horizon - 1, z, :] = resource_grid
    
    return result
# EVOLVE-BLOCK-END
