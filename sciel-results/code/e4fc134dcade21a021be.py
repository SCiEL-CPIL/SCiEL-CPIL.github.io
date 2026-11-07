import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    
    # Extract parameters
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Output grid
    m_out = resource_grid
    n_out = len(m_out)
    
    # Internal grids for high-resolution computation
    # Use a dense grid to approximate the continuous optimum
    # 500 points for m, 500 points for s (save fraction)
    m_int = np.geomspace(0.001, 40, 500)
    s_int = np.linspace(0.001, 1.0, 500)
    
    n_m = len(m_int)
    n_s = len(s_int)
    
    # Initialize policy and value arrays
    # Policy: shape (horizon, 2, n_m)
    # Value: shape (horizon, 2, n_m)
    policy_int = np.zeros((horizon, 2, n_m))
    value_int = np.zeros((horizon, 2, n_m))
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    # Backward induction
    # Terminal period t = horizon - 1
    t = horizon - 1
    for z in range(2):
        # At the last period, consume everything
        policy_int[t, z, :] = m_int
        value_int[t, z, :] = utility(m_int)
    
    # Previous periods
    for t in range(horizon - 2, -1, -1):
        for z in range(2):
            # Create meshgrid of m and s
            M_grid, S_grid = np.meshgrid(m_int, s_int, indexing='ij')
            C_grid = M_grid * S_grid
            
            # Calculate next period resources m'
            # m' = R * (m - c) + income[z']
            # We need to sum over z' weighted by transition probabilities
            # V_next(z', m')
            
            # Pre-compute expected continuation value for each (m, c)
            # This involves interpolating value_int[t+1, z', :] at m'
            
            cont_val = np.zeros_like(C_grid)
            
            for zp in range(2):
                prob = transition[z, zp]
                if prob == 0:
                    continue
                y_zp = income[zp]
                
                # m' for all (m, c) pairs
                m_prime = R * (M_grid - C_grid) + y_zp
                
                # Clip m_prime to the range of m_int for interpolation
                # np.interp requires xp to be increasing, which m_int is.
                # If m_prime is out of bounds, np.interp clamps to the end values.
                # This is a standard approximation for finite grids.
                
                # Interpolate value_int[t+1, zp, :] at m_prime
                # value_int[t+1, zp, :] is the value function for state zp at t+1
                v_next = np.interp(m_prime, m_int, value_int[t+1, zp, :])
                
                cont_val += prob * v_next
            
            # Total utility
            u_c = utility(C_grid)
            total_util = u_c + beta * cont_val
            
            # Find the best c for each m
            # total_util has shape (n_m, n_s)
            # We want the index of the max along axis 1 (s dimension)
            best_s_idx = np.argmax(total_util, axis=1)
            
            # Get the best c
            # C_grid is (n_m, n_s). We need to pick C_grid[i, best_s_idx[i]]
            best_c = C_grid[np.arange(n_m), best_s_idx]
            
            # Store policy and value
            policy_int[t, z, :] = best_c
            value_int[t, z, :] = total_util[np.arange(n_m), best_s_idx]
            
            # Time check
            if time.time() - start_time > 100:
                # If we run out of time, break and interpolate what we have
                # This is unlikely to happen with these grid sizes
                pass

    # Interpolate the internal policies to the output grid
    policy_out = np.zeros((horizon, 2, n_out))
    
    for t in range(horizon):
        for z in range(2):
            # policy_int[t, z, :] is defined on m_int
            # We want it on m_out
            # np.interp requires xp to be sorted, which m_int is.
            policy_out[t, z, :] = np.interp(m_out, m_int, policy_int[t, z, :])
            
            # Enforce constraints: 0 < c <= m + 1e-10
            # The interpolation should respect c <= m if the policy did, 
            # but numerical errors or grid boundaries might cause issues.
            # Also c must be > 0.
            policy_out[t, z, :] = np.maximum(policy_out[t, z, :], 1e-12)
            policy_out[t, z, :] = np.minimum(policy_out[t, z, :], m_out + 1e-10)
            
            # Special case for t = horizon - 1: c should be exactly m
            if t == horizon - 1:
                policy_out[t, z, :] = m_out

    return policy_out
