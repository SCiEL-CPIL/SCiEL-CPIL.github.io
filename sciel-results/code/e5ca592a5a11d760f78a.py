import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    Z = len(income)
    M_out = len(resource_grid)

    # Utility function
    def utility(c):
        c_safe = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c_safe)
        else:
            return (c_safe ** (1 - gamma)) / (1 - gamma)

    # Internal grid for value function: denser and wider
    # Range: 0.001 to 45 (to cover R*40 + max_income)
    m_internal = np.geomspace(0.001, 45, 501)
    M_int = len(m_internal)
    
    # Initialize V for t = horizon - 1
    # V[t, z, m] is the value. Since utility is state-independent (only m matters), 
    # V is same for all z at the last period? 
    # Wait, V_{T}(m) = U(m). It does not depend on z.
    # But for t < T, V_t(m, z) depends on z because the transition to z' depends on z.
    
    V_next = utility(m_internal)  # Shape (M_int,)
    
    # Storage for final consumption policy on the original grid
    consumption = np.zeros((horizon, Z, M_out))
    
    # We need to store V for each t to compute the policy for t-1?
    # Yes, backward induction.
    # We will compute V_t and Policy_t for t = T-1 down to 0.
    
    # Pre-compute interpolation of V_next from m_internal to any m_prime
    # We will use np.interp which is fast.
    
    # Candidate c grid for optimization
    # c must be in (0, m). 
    # We will use a grid of fractions f in [0.001, 1.0] to ensure c <= m.
    # But c cannot be 0. 
    # Let's use N_c = 200 points for c search.
    N_c = 200
    f_grid = np.linspace(0.001, 1.0, N_c)
    
    # We need to store V for the next period to compute the current one.
    # V_next is shape (Z, M_int). 
    # At t=T-1, V is independent of z, so V_next[z, :] = utility(m_internal) for all z.
    V_next_full = np.tile(V_next[np.newaxis, :], (Z, 1))  # Shape (Z, M_int)
    
    # Store policies for all t, z on the internal grid first, then interpolate to output grid
    # Or just compute on output grid directly? 
    # The "Knowledge" says: "interpolating the policy onto the 101-point resource_grid".
    # So we compute on internal grid, then interpolate.
    
    policy_internal = np.zeros((horizon, Z, M_int))
    
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 5.0:
            break
            
        if t == horizon - 1:
            # Policy is c = m
            policy_internal[t, :, :] = m_internal[np.newaxis, np.newaxis, :]
            # V_next_full is already set to utility(m_internal) for all z
            # But wait, for the next iteration (t-1), we need V_t.
            # V_t(m, z) = U(m) for all z.
            # So V_next_full for the next step (which is V_t) is just utility(m_internal) tiled.
            # It is already correct.
            continue
            
        # Compute V_t and Policy_t
        # V_t(z, m) = max_c [ U(c) + beta * sum_{z'} P(z'|z) V_{t+1}(z', m') ]
        # m' = R(m-c) + income[z']
        
        V_t = np.zeros((Z, M_int))
        
        for z in range(Z):
            # We want to find optimal c for each m in m_internal
            # Vectorize over m_internal (axis 0) and c_candidates (axis 1)
            
            # m is shape (M_int, 1)
            m_col = m_internal[:, np.newaxis]
            
            # c candidates: c = f * m
            # f is shape (1, N_c)
            # c is shape (M_int, N_c)
            c_candidates = m_col * f_grid[np.newaxis, :]
            
            # Ensure c > 0
            c_candidates = np.maximum(c_candidates, 1e-12)
            
            # Compute m' for each m, c, and z'
            # m_save = R * (m - c)
            # m' = m_save + income[z']
            
            # m_save shape (M_int, N_c)
            m_save = R * (m_col - c_candidates)
            
            # income shape (Z,)
            # We need E[ V_next(z', m') ]
            # V_next is shape (Z, M_int)
            
            # We can loop over z' here, as Z is small (2)
            expected_V = np.zeros((M_int, N_c))
            
            for zp in range(Z):
                prob = transition[z, zp]
                if prob == 0:
                    continue
                
                m_prime = m_save + income[zp]  # Shape (M_int, N_c)
                
                # Interpolate V_next[zp, :] at m_prime
                # np.interp expects 1D. We can flatten, interp, reshape.
                # Or use a loop over M_int? No, vectorize.
                # np.interp(fa, xp, fp)
                # fa: m_prime.ravel()
                # xp: m_internal
                # fp: V_next[zp, :]
                
                m_prime_flat = m_prime.ravel()
                V_interp = np.interp(m_prime_flat, m_internal, V_next[zp, :])
                V_interp = V_interp.reshape((M_int, N_c))
                
                expected_V += prob * V_interp
            
            # Objective = U(c) + beta * expected_V
            obj = utility(c_candidates) + beta * expected_V
            
            # Find optimal c for each m
            idx_max = np.argmax(obj, axis=1)
            
            # Extract optimal c
            # c_candidates is (M_int, N_c). idx_max is (M_int,)
            # Use advanced indexing
            c_opt = c_candidates[np.arange(M_int), idx_max]
            
            # Store policy on internal grid
            policy_internal[t, z, :] = c_opt
            
            # Store V_t
            V_t[z, :] = obj[np.arange(M_int), idx_max]
        
        # Set V_next_full for the next iteration (t-1)
        V_next_full = V_t
        
    # Now interpolate policy_internal (shape horizon, Z, M_int) to resource_grid (M_out)
    # policy_internal[t, z, :] is defined on m_internal.
    # We want policy on resource_grid.
    
    for t in range(horizon):
        for z in range(Z):
            # np.interp expects xp increasing. m_internal is increasing.
            # resource_grid is increasing.
            consumption[t, z, :] = np.interp(resource_grid, m_internal, policy_internal[t, z, :])
            
            # Enforce constraints: 0 < c <= m + 1e-10
            # c <= m
            consumption[t, z, :] = np.minimum(consumption[t, z, :], resource_grid + 1e-10)
            # c > 0
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-10)
            
            # Special case for last period t=horizon-1: c must be m
            if t == horizon - 1:
                consumption[t, z, :] = resource_grid
    
    return consumption
