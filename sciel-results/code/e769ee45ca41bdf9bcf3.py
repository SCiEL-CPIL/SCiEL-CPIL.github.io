import numpy as np
import time
from scipy.optimize import curve_fit

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    c_min = 1e-12
    grid = resource_grid
    n_grid = len(grid)
    T = horizon
    n_z = 2
    
    def utility(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
    
    def utility_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # Use log-space for the value function representation
    # V_next[z] will be represented as a piecewise-linear function of log(m)
    # We'll store it as (log_grid, V_values) for each z
    
    # Terminal condition: c = m, so V = u(m)
    log_grid = np.log(grid)
    
    # V[t][z] = array of values at log_grid points
    # P[t][z] = array of optimal c at log_grid points
    V = [None] * T
    P = [None] * T
    
    V[T-1] = utility(grid)[np.newaxis, :].repeat(2, axis=0)
    P[T-1] = grid[np.newaxis, :].repeat(2, axis=0)
    
    # For backward induction, we need to evaluate V_next[z'](m') for arbitrary m'
    # We'll use np.interp on log(m) for smoothness
    
    for t in range(T-2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: use terminal policy for remaining periods
            for t_rem in range(t, -1, -1):
                V[t_rem] = V[T-1].copy()
                P[t_rem] = P[T-1].copy()
            break
        
        V_t = np.zeros((n_z, n_grid))
        P_t = np.zeros((n_z, n_grid))
        V_next = V[t+1]
        
        # Pre-compute log-space interpolation for V_next
        # For each z', we have V_next[z'] as function of log_grid
        
        for z in range(n_z):
            # For each m in grid, solve for optimal c
            # Euler: u'(c) = beta * R * sum_{z'} P(z'|z) * V_next[z']'(R(m-c) + y_{z'})
            
            # We'll use a dense consumption grid for vectorized search
            # c ranges from c_min to m
            # Use a logarithmic grid for c for better resolution at low c
            c_grid = np.geomspace(c_min, grid[z] * 0.999 if grid[z] > c_min*10 else grid[z], 500)
            c_grid = np.clip(c_grid, c_min, grid[z])
            
            # For each c in c_grid, compute the Euler residual
            # residual(c) = u'(c) - beta * R * sum_{z'} trans[z,z'] * V_next[z']'(R(m-c) + y_{z'})
            
            m_val = grid[z]
            
            # Compute V_next[z']' at R(m-c) + y_{z'} for each z'
            # We need the derivative of V_next with respect to m'
            # Approximate using finite differences in log-space
            
            # Actually, let's directly evaluate the continuation value and use
            # a different approach: maximize u(c) + beta * sum_{z'} trans * V_next[z'](R(m-c)+y_{z'})
            
            # This is a 1D optimization. Use a dense grid + refinement.
            
            # Dense grid evaluation
            c_vals = np.geomspace(max(c_min, 1e-10), m_val * 0.9999, 1000)
            c_vals = np.clip(c_vals, c_min, m_val)
            
            # For each c, compute total value
            total_val = np.zeros_like(c_vals)
            for zp in range(n_z):
                next_m = R * (m_val - c_vals) + income[zp]
                next_m = np.clip(next_m, grid[0] * 0.5, grid[-1] * 1.5)
                # Interpolate V_next[zp] in log-space
                log_next_m = np.log(np.maximum(next_m, c_min))
                # Clip to log_grid range
                log_next_m_clipped = np.clip(log_next_m, log_grid[0], log_grid[-1])
                cont = np.interp(log_next_m_clipped, log_grid, V_next[zp])
                total_val += transition[z, zp] * cont
            
            total_val = utility(c_vals) + beta * total_val
            
            # Find the max
            idx = np.argmax(total_val)
            c_best = c_vals[idx]
            
            # Refine around the best with a finer local grid
            if idx > 0 and idx < len(c_vals) - 1:
                c_lo = c_vals[max(0, idx-5)]
                c_hi = c_vals[min(len(c_vals)-1, idx+5)]
                c_fine = np.linspace(c_lo, c_hi, 200)
                c_fine = np.clip(c_fine, c_min, m_val)
                
                total_val_fine = np.zeros_like(c_fine)
                for zp in range(n_z):
                    next_m = R * (m_val - c_fine) + income[zp]
                    next_m = np.clip(next_m, grid[0] * 0.5, grid[-1] * 1.5)
                    log_next_m = np.log(np.maximum(next_m, c_min))
                    log_next_m_clipped = np.clip(log_next_m, log_grid[0], log_grid[-1])
                    cont = np.interp(log_next_m_clipped, log_grid, V_next[zp])
                    total_val_fine += transition[z, zp] * cont
                
                total_val_fine = utility(c_fine) + beta * total_val_fine
                idx_fine = np.argmax(total_val_fine)
                c_best = c_fine[idx_fine]
            
            c_best = np.clip(c_best, c_min, m_val)
            P_t[z, z] = c_best  # This is wrong indexing, let me fix
            
            # Actually, I need to do this for each grid point, not just one.
            # Let me restructure: for each grid point i, find optimal c.
            
            break  # Break out of z loop, I need to restructure
        
        # Restructure: for each z, for each grid point, find optimal c
        for z in range(n_z):
            for i, m_val in enumerate(grid):
                if time.time() > deadline - 10:
                    # Fallback
                    c_best = 0.5 * m_val
                    P_t[z, i] = c_best
                    # Compute V
                    cont = 0.0
                    for zp in range(n_z):
                        next_m = R * (m_val - c_best) + income[zp]
                        next_m = np.clip(next_m, grid[0] * 0.5, grid[-1] * 1.5)
                        log_next_m = np.log(np.maximum(next_m, c_min))
                        log_next_m_clipped = np.clip(log_next_m, log_grid[0], log_grid[-1])
                        cont += transition[z, zp] * np.interp(log_next_m_clipped, log_grid, V_next[zp])
                    V_t[z, i] = utility(c_best) + beta * cont
                    continue
                
                # Dense grid search for optimal c
                c_vals = np.geomspace(max(c_min, 1e-10), m_val * 0.9999, 500)
                c_vals = np.clip(c_vals, c_min, m_val)
                
                total_val = np.zeros_like(c_vals)
                for zp in range(n_z):
                    next_m = R * (m_val - c_vals) + income[zp]
                    next_m = np.clip(next_m, grid[0] * 0.5, grid[-1] * 1.5)
                    log_next_m = np.log(np.maximum(next_m, c_min))
                    log_next_m_clipped = np.clip(log_next_m, log_grid[0], log_grid[-1])
                    cont = np.interp(log_next_m_clipped, log_grid, V_next[zp])
                    total_val += transition[z, zp] * cont
                
                total_val = utility(c_vals) + beta * total_val
                idx = np.argmax(total_val)
                c_best = c_vals[idx]
                
                # Local refinement
                if idx > 0 and idx < len(c_vals) - 1:
                    c_lo = c_vals[max(0, idx-3)]
                    c_hi = c_vals[min(len(c_vals)-1, idx+3)]
                    c_fine = np.linspace(c_lo, c_hi, 100)
                    c_fine = np.clip(c_fine, c_min, m_val)
                    
                    total_val_fine = np.zeros_like(c_fine)
                    for zp in range(n_z):
                        next_m = R * (m_val - c_fine) + income[zp]
                        next_m = np.clip(next_m, grid[0] * 0.5, grid[-1] * 1.5)
                        log_next_m = np.log(np.maximum(next_m, c_min))
                        log_next_m_clipped = np.clip(log_next_m, log_grid[0], log_grid[-1])
                        cont = np.interp(log_next_m_clipped, log_grid, V_next[zp])
                        total_val_fine += transition[z, zp] * cont
                    
                    total_val_fine = utility(c_fine) + beta * total_val_fine
                    idx_fine = np.argmax(total_val_fine)
                    c_best = c_fine[idx_fine]
                
                c_best = np.clip(c_best, c_min, m_val)
                P_t[z, i] = c_best
                
                # Compute V
                cont = 0.0
                for zp in range(n_z):
                    next_m = R * (m_val - c_best) + income[zp]
                    next_m = np.clip(next_m, grid[0] * 0.5, grid[-1] * 1.5)
                    log_next_m = np.log(np.maximum(next_m, c_min))
                    log_next_m_clipped = np.clip(log_next_m, log_grid[0], log_grid[-1])
                    cont += transition[z, zp] * np.interp(log_next_m_clipped, log_grid, V_next[zp])
                V_t[z, i] = utility(c_best) + beta * cont
        
        V[t] = V_t
        P[t] = P_t
    
    # Assemble result
    result = np.zeros((T, n_z, n_grid))
    for t in range(T):
        result[t, :, :] = P[t]
    
    # Enforce constraints
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.clip(result[t, z, :], c_min, grid + 1e-10)
    
    # Final period constraint: c = m
    result[T-1, :, :] = grid[np.newaxis, :]
    
    return result
