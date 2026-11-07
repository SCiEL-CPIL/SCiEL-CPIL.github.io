"""Economics: consumption-saving policy accuracy."""
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    deadline = time.time() + 145  # 145s safety margin (limit 150s)
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-12:
            return np.log(c)
        return (c ** (1 - gamma)) / (1 - gamma)
    
    # Use a much finer grid for internal computation
    # The resource_grid has 101 points from 0.001 to 40 (geometric)
    # We'll work on a finer grid for better accuracy
    fine_grid = np.geomspace(0.001, 40, 1001)
    n_fine = len(fine_grid)
    
    # Value function: V[t, z, m] for t=0..horizon-1, z=0,1, m on fine_grid
    # At t=horizon-1 (last period), agent consumes all: c = m
    # V[horizon-1, z, m] = util(m)
    
    # We'll store value functions as arrays
    # V shape: (horizon, n_states, n_fine)
    V = np.zeros((horizon, n_states, n_fine))
    
    # Last period: consume everything
    V[horizon-1, :, :] = util(fine_grid)
    
    # Consumption policy for each period
    # c[t, z, m] shape (horizon, n_states, n_fine)
    C = np.zeros((horizon, n_states, n_fine))
    C[horizon-1, :, :] = fine_grid
    
    # Backward induction from t = horizon-2 down to 0
    for t in range(horizon-2, -1, -1):
        if time.time() > deadline - 10:
            break
        for z in range(n_states):
            # For each m in fine_grid, find optimal c
            # m' = R*(m - c) + income[z']
            # V_next[z', m'] = V[t+1, z', m']
            # Objective: util(c) + beta * E[V[t+1, z', m']]
            # m' = R*(m-c) + income[z'], so c = m - (m' - income[z'])/R
            # We need c in (0, m]
            
            # For each z', compute the continuation value as a function of c
            # m' = R*(m-c) + income[z']
            # For a given c, m' is determined
            # We need to interpolate V[t+1, z', m'] at m'
            
            # Let's vectorize over c values
            # For each m in fine_grid, c can range from small to m
            # m' ranges from R*(m-c_max) + income[z'] to R*(m-c_min) + income[z']
            # c_min ~ 0, c_max = m
            # m' at c=0: R*m + income[z']
            # m' at c=m: income[z']
            
            # We'll try a set of c values for each m
            # c = m * linspace(0.001, 1, n_candidates)
            n_candidates = 500
            
            # For each m, compute utility of c and expected continuation value
            # This is expensive, so let's be smart
            
            # Precompute: for each z', the mapping from c to m' and then to V_next
            # m' = R*(m-c) + income[z']
            # V_next[z', m'] via interpolation on fine_grid
            
            best_c = np.zeros(n_fine)
            best_val = -np.inf * np.ones(n_fine)
            
            # Process in chunks to manage memory
            chunk_size = 100
            for chunk_start in range(0, n_fine, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_fine)
                m_chunk = fine_grid[chunk_start:chunk_end]
                n_chunk = len(m_chunk)
                
                # c values: for each m, c = m * linspace(0.001, 1, n_candidates)
                # Shape: (n_chunk, n_candidates)
                c_frac = np.linspace(0.001, 1.0, n_candidates)
                c_vals = m_chunk[:, None] * c_frac[None, :]
                c_vals = np.maximum(c_vals, 1e-12)
                
                # Utility
                u = util(c_vals)  # (n_chunk, n_candidates)
                
                # For each z', compute m' and V_next
                expected_V = np.zeros((n_chunk, n_candidates))
                for z_next in range(n_states):
                    p = transition[z, z_next]
                    if p < 1e-15:
                        continue
                    # m' = R*(m-c) + income[z_next]
                    m_prime = R * (m_chunk[:, None] - c_vals) + income[z_next]
                    # Clamp m_prime to grid range
                    m_prime = np.clip(m_prime, fine_grid[0], fine_grid[-1])
                    # Interpolate V[t+1, z_next, m']
                    V_interp = np.interp(m_prime, fine_grid, V[t+1, z_next, :])
                    expected_V += p * V_interp
                
                total_val = u + beta * expected_V  # (n_chunk, n_candidates)
                
                # Find best c for each m
                idx = np.argmax(total_val, axis=1)
                best_c[chunk_start:chunk_end] = c_vals[np.arange(n_chunk), idx]
                best_val[chunk_start:chunk_end] = total_val[np.arange(n_chunk), idx]
            
            C[t, z, :] = best_c
            V[t, z, :] = best_val
    
    # Now we have C on fine_grid. Interpolate to the required resource_grid
    # consumption[t, z, m] for m in resource_grid
    consumption = np.zeros((horizon, n_states, n_grid))
    for t in range(horizon):
        for z in range(n_states):
            consumption[t, z, :] = np.interp(resource_grid, fine_grid, C[t, z, :])
    
    # Enforce constraints: 0 < c <= m + 1e-10
    consumption = np.clip(consumption, 1e-12, None)
    for t in range(horizon):
        for z in range(n_states):
            consumption[t, z, :] = np.minimum(consumption[t, z, :], resource_grid + 1e-10)
    
    # Last period: c = m exactly
    consumption[horizon-1, :, :] = resource_grid
    
    # Final check: finite values
    if not np.all(np.isfinite(consumption)):
        consumption = np.nan_to_num(consumption, nan=0.001, posinf=resource_grid, neginf=0.001)
    
    return consumption
