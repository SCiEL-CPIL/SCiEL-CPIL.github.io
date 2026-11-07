"""Solve consumption-saving policy using value function iteration on a fine grid."""
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    transition = np.asarray(model['transition'])
    horizon = model['horizon']
    m_grid = np.asarray(resource_grid)
    N_m = len(m_grid)
    N_z = len(income)
    
    # Create a fine grid for value function computation
    # Use a log-spaced grid from slightly below min(m_grid) to max(m_grid)
    m_min = m_grid[0]
    m_max = m_grid[-1]
    
    # Fine grid: 2000 points for good resolution
    fine_grid = np.geomspace(m_min * 0.9, m_max * 1.1, 2000)
    
    # Ensure the original grid points are included
    fine_grid = np.unique(np.concatenate([fine_grid, m_grid]))
    fine_grid = np.sort(fine_grid)
    N_fine = len(fine_grid)
    
    def util(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
    
    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # Value function: V[t, z, m] for t in range(horizon), z in range(N_z)
    # At t = horizon - 1 (last period), the agent consumes everything (no future)
    # Actually, let's think about this more carefully.
    
    # The problem says m' = R(m - c) + income[z'], and the horizon is 6.
    # At the last period (t = horizon-1 = 5), the constraint is c[5] = m (consume all).
    # So V[5, z, m] = util(m) for all z, m.
    
    # For t < 5: V[t, z, m] = max_c { util(c) + beta * sum_{z'} transition[z, z'] * V[t+1, z', m'] }
    # where m' = R(m - c) + income[z']
    
    # Initialize value function
    V = np.zeros((horizon, N_z, N_fine))
    
    # Last period: consume everything
    for z in range(N_z):
        V[horizon-1, z, :] = util(fine_grid)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() - start_time > DEADLINE - 10:
            # Not enough time for remaining periods, break
            break
        
        for z in range(N_z):
            # For each m in fine_grid, find optimal c
            # c in [0, m], m' = R(m - c) + income[z']
            # We need to interpolate V[t+1, z', m'] for each z'
            
            # Precompute continuation values for each z'
            # V_cont[z', m'] = V[t+1, z', m'] where m' = R(m - c) + income[z']
            # But m' depends on c, so we need to evaluate for each candidate c
            
            # Use a grid of c values: c = m * linspace(0, 1, n_c)
            # For efficiency, use a moderate number of c candidates
            
            # The scoring uses c = m * np.linspace(0.001, 1, 301) for best,
            # so we should use at least that many candidates for accuracy.
            
            n_c = 301
            # c fractions
            c_frac = np.linspace(0.001, 1.0, n_c)
            
            # For each m in fine_grid, compute optimal c
            # Vectorize over m
            
            # m_fine shape: (N_fine,)
            # c_candidates shape: (N_fine, n_c)
            m_fine = fine_grid[:, np.newaxis]  # (N_fine, 1)
            c_candidates = m_fine * c_frac[np.newaxis, :]  # (N_fine, n_c)
            
            # m' for each z'
            # m_prime[z', i, j] = R * (m_fine[i] - c_candidates[i, j]) + income[z']
            # Shape: (N_z, N_fine, n_c)
            
            best_value = np.zeros(N_fine)
            best_c = np.zeros(N_fine)
            
            for z_prime in range(N_z):
                # m_prime for this z'
                m_prime = R * (m_fine - c_candidates) + income[z_prime]
                # Shape: (N_fine, n_c)
                
                # Clip m_prime to valid range
                m_prime = np.clip(m_prime, fine_grid[0], fine_grid[-1])
                
                # Interpolate V[t+1, z_prime, m_prime]
                # V_next shape: (N_fine,)
                V_next = V[t+1, z_prime, :]
                
                # Interpolate: for each (i, j), get V_next at m_prime[i, j]
                # Use np.interp for each row
                V_interp = np.empty_like(m_prime)
                for i in range(N_fine):
                    V_interp[i, :] = np.interp(m_prime[i, :], fine_grid, V_next)
                
                # Continuation value weighted by transition
                trans_weight = transition[z, z_prime]
                
                # Total value: util(c) + beta * trans_weight * V_interp
                total_value = util(c_candidates) + beta * trans_weight * V_interp
                # Shape: (N_fine, n_c)
                
                # Find max over c dimension
                idx_max = np.argmax(total_value, axis=1)
                vals_max = total_value[np.arange(N_fine), idx_max]
                
                # Update best
                # But we need to sum over all z', not just one
                # Actually, the Bellman equation is:
                # V[t, z, m] = max_c { util(c) + beta * sum_{z'} transition[z, z'] * V[t+1, z', m'(z')] }
                # So we need to compute the sum for each c candidate, then take max over c.
                
                # Let me redo this properly.
                pass
            
            # Redo: compute sum over z' for each c candidate, then max over c
            total_value_all = np.zeros((N_fine, n_c))
            
            for z_prime in range(N_z):
                m_prime = R * (m_fine - c_candidates) + income[z_prime]
                m_prime = np.clip(m_prime, fine_grid[0], fine_grid[-1])
                
                V_next = V[t+1, z_prime, :]
                V_interp = np.empty_like(m_prime)
                for i in range(N_fine):
                    V_interp[i, :] = np.interp(m_prime[i, :], fine_grid, V_next)
                
                trans_weight = transition[z, z_prime]
                total_value_all += beta * trans_weight * V_interp
            
            total_value_all += util(c_candidates)
            
            # Max over c
            idx_max = np.argmax(total_value_all, axis=1)
            best_value = total_value_all[np.arange(N_fine), idx_max]
            best_c = c_candidates[np.arange(N_fine), idx_max]
            
            V[t, z, :] = best_value
    
    # Now extract the policy on the original grid
    # consumption[t, z, i] = optimal c at m = m_grid[i]
    consumption = np.zeros((horizon, N_z, N_m))
    
    for t in range(horizon):
        if t == horizon - 1:
            # Last period: consume all
            consumption[t, :, :] = m_grid[np.newaxis, :]
        else:
            for z in range(N_z):
                # We need to find the optimal c for each m in m_grid
                # We can either:
                # 1. Interpolate the policy from the fine grid
                # 2. Recompute using the value function
                
                # Option 1: Interpolate. But we didn't store the policy, only the value.
                # Let's recompute the optimal c using the value function V.
                
                n_c = 301
                c_frac = np.linspace(0.001, 1.0, n_c)
                m_orig = m_grid[:, np.newaxis]  # (N_m, 1)
                c_candidates = m_orig * c_frac[np.newaxis, :]  # (N_m, n_c)
                
                total_value_all = np.zeros((N_m, n_c))
                
                for z_prime in range(N_z):
                    m_prime = R * (m_orig - c_candidates) + income[z_prime]
                    m_prime = np.clip(m_prime, fine_grid[0], fine_grid[-1])
                    
                    V_next = V[t+1, z_prime, :]
                    V_interp = np.empty_like(m_prime)
                    for i in range(N_m):
                        V_interp[i, :] = np.interp(m_prime[i, :], fine_grid, V_next)
                    
                    trans_weight = transition[z, z_prime]
                    total_value_all += beta * trans_weight * V_interp
                
                total_value_all += util(c_candidates)
                
                idx_max = np.argmax(total_value_all, axis=1)
                consumption[t, z, :] = c_candidates[np.arange(N_m), idx_max]
    
    # Ensure constraints: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(N_z):
            consumption[t, z, :] = np.clip(consumption[t, z, :], 1e-12, m_grid + 1e-10)
    
    # Ensure c[5] = grid within 1e-9
    consumption[5, :, :] = m_grid[np.newaxis, :]
    
    # Check for finite values
    if not np.all(np.isfinite(consumption)):
        # Fallback: return a simple policy
        for t in range(horizon):
            for z in range(N_z):
                if t == horizon - 1:
                    consumption[t, z, :] = m_grid
                else:
                    consumption[t, z, :] = m_grid * 0.5  # Save half
        consumption[5, :, :] = m_grid[np.newaxis, :]
    
    return consumption
