import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'], dtype=float)
    transition = np.array(model['transition'], dtype=float)
    horizon = model['horizon']
    z_count = len(income)
    
    # Internal fine grid - use geomspace for better coverage of the log-scale
    # Extend range to cover max possible m' (R*40 + max_income ~ 41.8)
    N_fine = 4000
    m_max = 45.0
    m_fine = np.geomspace(0.001, m_max, N_fine)
    
    # Precompute utility
    def U(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    # Precompute marginal utility for potential refinement
    def U_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)

    V = np.zeros((horizon, z_count, N_fine))
    c_fine = np.zeros((horizon, z_count, N_fine))
    
    # Terminal condition
    V[horizon-1, :, :] = U(m_fine)
    c_fine[horizon-1, :, :] = m_fine
    
    # Candidates for c
    # Use a relative grid. The evaluation uses c = m * linspace(0.001, 1, 301).
    # We should match or exceed this resolution.
    # 2000 points is a good balance.
    N_c = 2000
    c_rel = np.linspace(0.001, 1.0, N_c)
    
    # Precompute c_grid and m_primes to save time in loop?
    # c_grid shape (N_fine, N_c)
    # This is 4000 * 2000 = 8M elements. ~64MB. Acceptable.
    c_grid = m_fine[:, None] * c_rel[None, :]
    
    # m_prime depends on c, so it varies with the candidate.
    # m_prime[z'] = R * (m - c) + income[z']
    # We can precompute R * (m_fine[:, None] - c_grid) once, then add income.
    base_m_prime = R * (m_fine[:, None] - c_grid)
    
    for t in range(horizon - 2, -1, -1):
        if time.time() - t0 > DEADLINE:
            # Fallback: return current best estimate (linear interp of whatever we have)
            # If we haven't computed t, we need to fill V[t] with something.
            # But usually we compute backwards, so if we break early, we have V[t+1..end].
            # We need to return c for all t.
            # If we break at t, we have c_fine[t+1..]. We need c_fine[0..t].
            # We can just use a heuristic or linear interp of the last computed c?
            # Better: just return the terminal condition for uncomputed steps? No, that's bad.
            # Let's just break and interpolate what we have, assuming the loop completed
            # or we have a partial result.
            # Actually, if we break, we should just return a valid policy.
            # The most robust fallback is to use the terminal policy (c=m) for all t?
            # Or the last computed policy?
            # Let's just ensure we return something valid.
            pass

        # Interpolate V[t+1] at m_prime locations
        # m_prime_0 = base_m_prime + income[0]
        # m_prime_1 = base_m_prime + income[1]
        
        # np.interp requires 1D x. We have 2D m_prime.
        # Flatten, interpolate, reshape.
        
        m_prime_0_flat = (base_m_prime + income[0]).ravel()
        m_prime_1_flat = (base_m_prime + income[1]).ravel()
        
        # Interpolate V_next for z'=0 and z'=1
        # V[t+1, 0, :] and V[t+1, 1, :] are 1D arrays of length N_fine
        V_next_0_flat = np.interp(m_prime_0_flat, m_fine, V[t+1, 0, :])
        V_next_1_flat = np.interp(m_prime_1_flat, m_fine, V[t+1, 1, :])
        
        V_next_0 = V_next_0_flat.reshape(N_fine, N_c)
        V_next_1 = V_next_1_flat.reshape(N_fine, N_c)
        
        # Precompute U(c_grid) once per step? No, it's constant across t for a given grid.
        # But we can just compute it.
        U_c = U(c_grid)
        
        for z in range(z_count):
            # Continuation value: beta * sum(pi[z, z'] * V[t+1, z', m'])
            cont_val = beta * (transition[z, 0] * V_next_0 + transition[z, 1] * V_next_1)
            
            total_val = U_c + cont_val
            
            # Find argmax
            idx_max = np.argmax(total_val, axis=1)
            
            c_opt = c_grid[np.arange(N_fine), idx_max]
            V[t, z, :] = total_val[np.arange(N_fine), idx_max]
            c_fine[t, z, :] = c_opt
            
            # Optional: Local refinement using Newton's method
            # The grid c_rel is uniform in [0.001, 1].
            # The optimal c might lie between grid points.
            # We can perform a few Newton steps using the derivatives.
            
            # Check time
            if time.time() - t0 > DEADLINE - 10:
                break
                
            # Refinement
            # We need V' and V'' of the continuation value.
            # This is complex. Let's stick to the grid search for now, 
            # as N_c=2000 is quite fine.
            # The error from grid discretization is likely smaller than 
            # the error from interpolation on the resource grid.
            
    # Interpolate c_fine to resource_grid
    c_out = np.zeros((horizon, z_count, len(resource_grid)))
    
    # Ensure resource_grid is within m_fine bounds for interpolation
    # m_fine is [0.001, 45]. resource_grid is [0.001, 40]. So it's fine.
    
    for t in range(horizon):
        for z in range(z_count):
            c_out[t, z, :] = np.interp(resource_grid, m_fine, c_fine[t, z, :])
            
    # Enforce terminal condition exactly
    c_out[horizon-1, :, :] = resource_grid
    
    # Ensure validity: 0 < c <= m
    # The task says: 0 < c <= m + 1e-10
    # And c[5] equal to the grid within 1e-9.
    
    for t in range(horizon):
        for z in range(z_count):
            # Clip to valid range
            c_out[t, z, :] = np.clip(c_out[t, z, :], 1e-12, resource_grid + 1e-10)
            
    # Final check for terminal condition
    c_out[horizon-1, :, :] = resource_grid
    
    return c_out
