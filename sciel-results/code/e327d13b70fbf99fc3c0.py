import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = len(income)
    
    if abs(gamma - 1.0) < 1e-10:
        def util(c):
            c = np.maximum(c, 1e-12)
            return np.log(c)
        def util_prime(c):
            c = np.maximum(c, 1e-12)
            return 1.0 / c
    else:
        def util(c):
            c = np.maximum(c, 1e-12)
            return (c ** (1 - gamma) - 1) / (1 - gamma)
        def util_prime(c):
            c = np.maximum(c, 1e-12)
            return c ** (-gamma)
    
    c_policy = np.zeros((horizon, n_z, n_m))
    c_policy[horizon - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    V_next = np.tile(util(m_grid), (n_z, 1))
    Vm_next = np.tile(util_prime(m_grid), (n_z, 1))
    
    trans = transition.copy()
    active_zp = [zp for zp in range(n_z) if trans[0, zp] > 1e-10 or trans[1, zp] > 1e-10]
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            for tt in range(t, -1, -1):
                for z in range(n_z):
                    c_policy[tt, z, :] = m_grid
            break
        
        V_curr = np.zeros((n_z, n_m))
        Vm_curr = np.zeros((n_z, n_m))
        
        for z in range(n_z):
            # Rough initial guess
            m_next_mid = R * (m_grid - m_grid * 0.5) + income[np.newaxis, :]
            m_next_mid = np.clip(m_next_mid, m_grid[0], m_grid[-1])
            avg_Vm = np.zeros(n_m)
            for zp in range(n_z):
                if trans[z, zp] < 1e-10:
                    continue
                avg_Vm += trans[z, zp] * np.interp(m_next_mid[:, zp], m_grid, Vm_next[zp])
            
            if abs(gamma - 1.0) < 1e-10:
                c = 1.0 / (beta * R * np.maximum(avg_Vm, 1e-12))
            else:
                c = (beta * R * np.maximum(avg_Vm, 1e-12)) ** (-1.0 / gamma)
            c = np.clip(c, 1e-8, m_grid)
            
            # Vectorized Newton
            c_lo = np.full(n_m, 1e-8)
            c_hi = m_grid.copy()
            
            for it in range(30):
                if time.time() > deadline - 2:
                    break
                
                u_p = util_prime(c)
                m_next = R * (m_grid[np.newaxis, :] - c[:, np.newaxis]) + income[np.newaxis, :]
                m_next = np.clip(m_next, m_grid[0], m_grid[-1])
                
                E_Vm = np.zeros(n_m)
                for zp in range(n_z):
                    if trans[z, zp] < 1e-10:
                        continue
                    E_Vm += trans[z, zp] * np.interp(m_next[:, zp], m_grid, Vm_next[zp])
                
                f = u_p - beta * R * E_Vm
                
                if np.max(np.abs(f)) < 1e-12:
                    break
                
                # Numerical derivative
                h = np.maximum(1e-8, c * 1e-6)
                c_plus = np.minimum(c + h, c_hi)
                c_minus = np.maximum(c - h, c_lo)
                
                u_p_plus = util_prime(c_plus)
                m_next_plus = R * (m_grid[np.newaxis, :] - c_plus[:, np.newaxis]) + income[np.newaxis, :]
                m_next_plus = np.clip(m_next_plus, m_grid[0], m_grid[-1])
                E_Vm_plus = np.zeros(n_m)
                for zp in range(n_z):
                    if trans[z, zp] < 1e-10:
                        continue
                    E_Vm_plus += trans[z, zp] * np.interp(m_next_plus[:, zp], m_grid, Vm_next[zp])
                f_plus = u_p_plus - beta * R * E_Vm_plus
                
                u_p_minus = util_prime(c_minus)
                m_next_minus = R * (m_grid[np.newaxis, :] - c_minus[:, np.newaxis]) + income[np.newaxis, :]
                m_next_minus = np.clip(m_next_minus, m_grid[0], m_grid[-1])
                E_Vm_minus = np.zeros(n_m)
                for zp in range(n_z):
                    if trans[z, zp] < 1e-10:
                        continue
                    E_Vm_minus += trans[z, zp] * np.interp(m_next_minus[:, zp], m_grid, Vm_next[zp])
                f_minus = u_p_minus - beta * R * E_Vm_minus
                
                f_prime = (f_plus - f_minus) / (2 * h)
                
                # Newton step
                valid = np.abs(f_prime) > 1e-14
                c_new = c.copy()
                c_new[valid] = c[valid] - f[valid] / f_prime[valid]
                
                # Bisection safeguard
                c_new = np.clip(c_new, c_lo, c_hi)
                
                # Update bounds
                c_lo = np.where(f > 0, c, c_lo)
                c_hi = np.where(f < 0, c, c_hi)
                
                if np.max(np.abs(c_new - c)) < 1e-12 * np.max(1.0 + np.abs(c)):
                    c = c_new
                    break
                
                c = c_new
            
            c = np.clip(c, 1e-12, m_grid)
            c_policy[t, z, :] = c
            Vm_curr[z, :] = util_prime(c)
            
            m_next = R * (m_grid[np.newaxis, :] - c[:, np.newaxis]) + income[np.newaxis, :]
            m_next = np.clip(m_next, m_grid[0], m_grid[-1])
            
            V_val = util(c).copy()
            for zp in range(n_z):
                if trans[z, zp] < 1e-10:
                    continue
                V_val += beta * trans[z, zp] * np.interp(m_next[:, zp], m_grid, V_next[zp])
            
            V_curr[z, :] = V_val
        
        V_next = V_curr
        Vm_next = Vm_curr
    
    for t in range(horizon):
        for z in range(n_z):
            c_policy[t, z, :] = np.clip(c_policy[t, z, :], 1e-12, m_grid)
    
    c_policy[horizon - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    return c_policy
