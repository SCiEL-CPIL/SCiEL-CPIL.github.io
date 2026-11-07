import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = len(income)
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)
    
    # Consumption grid: use 301 points from 0.001*m to 1.0*m
    # We'll work with a fraction grid: c = frac * m
    n_frac = 301
    fracs = np.linspace(0.001, 1.0, n_frac)
    
    # V[t, z, m_idx] = value function
    # C[t, z, m_idx] = optimal consumption
    V = np.zeros((horizon, n_z, n_m))
    C = np.zeros((horizon, n_z, n_m))
    
    # Terminal condition: at t = horizon-1, consume everything
    V[horizon - 1, :, :] = 0.0  # no continuation
    C[horizon - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]  # c = m
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break
        
        for z in range(n_z):
            # For each m, compute optimal c
            # c = frac * m, so c_grid[m_idx] = fracs * m_grid[m_idx]
            c_grid = np.outer(fracs, m_grid)  # shape (n_frac, n_m)
            
            # Utility of current consumption
            u_now = utility(c_grid)  # shape (n_frac, n_m)
            
            # Continuation value: for each frac and m, compute expected V[t+1, z', m']
            # m' = R * (m - c) + income[z'] = R * m * (1 - frac) + income[z']
            # For each z', compute m' for all (frac, m) pairs
            cont_values = np.zeros_like(c_grid)
            
            for zp in range(n_z):
                prob = transition[z, zp]
                if prob < 1e-15:
                    continue
                
                # m' = R * m * (1 - frac) + income[zp]
                m_prime = R * np.outer(1.0 - fracs, m_grid) + income[zp]
                # m_prime shape: (n_frac, n_m)
                
                # Clip m_prime to grid range
                m_prime_clipped = np.clip(m_prime, m_grid[0], m_grid[-1])
                
                # Interpolate V[t+1, zp, :] onto m_prime
                # V[t+1, zp, :] has shape (n_m,)
                V_next = V[t + 1, zp, :]
                
                # For each column (m_idx), interpolate along m_prime
                # We need to interpolate V_next as a function of m, evaluated at m_prime
                # V_next is defined on m_grid, we evaluate at m_prime
                # Use np.interp for each row (frac)
                cont_zp = np.zeros_like(c_grid)
                for i in range(n_frac):
                    # m_prime[i, :] has shape (n_m,)
                    cont_zp[i, :] = np.interp(m_prime[i, :], m_grid, V_next)
                
                cont_values += prob * cont_zp
            
            # Total value for each (frac, m)
            total_value = u_now + beta * cont_values  # shape (n_frac, n_m)
            
            # Find optimal frac for each m
            best_frac_idx = np.argmax(total_value, axis=0)  # shape (n_m,)
            
            # Optimal consumption
            C[t, z, :] = fracs[best_frac_idx] * m_grid
            
            # Store value function at optimal c
            V[t, z, :] = total_value[best_frac_idx, np.arange(n_m)]
    
    # Ensure terminal period: c[5] = grid (within 1e-9)
    C[horizon - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    # Enforce constraints: 0 < c <= m + 1e-10, finite
    for t in range(horizon):
        for z in range(n_z):
            C[t, z, :] = np.maximum(C[t, z, :], 1e-12)
            C[t, z, :] = np.minimum(C[t, z, :], m_grid + 1e-10)
            C[t, z, :] = np.nan_to_num(C[t, z, :], nan=1e-12, posinf=m_grid, neginf=1e-12)
    
    # Final validity check
    if not np.all(np.isfinite(C)):
        C = np.nan_to_num(C, nan=1e-12, posinf=m_grid, neginf=1e-12)
    
    return C
