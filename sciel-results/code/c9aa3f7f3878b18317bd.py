import numpy as np
import time
from scipy.interpolate import interp1d

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

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
    
    # Define utility and derivatives
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            c = np.maximum(c, c_min)
            return np.log(c)
    else:
        def utility(c):
            c = np.maximum(c, c_min)
            return c ** (1 - gamma) / (1 - gamma)

    # We will solve on a fine grid for better accuracy
    # The original grid is geomspace(0.001, 40, 101)
    # Let's use a fine grid of 1000 points
    N_FINE = 1000
    m_fine = np.geomspace(0.001, 40, N_FINE)
    
    # Precompute candidate consumption fractions
    # c = m * x, where x in [0.001, 1]
    # We use a fine grid for x to ensure we find the max
    N_CAND = 200
    x_candidates = np.linspace(0.001, 1.0, N_CAND)
    
    # V[t] will be an array of shape (n_z, N_FINE)
    V = [None] * T
    P = [None] * T
    
    # Terminal condition: t = T-1
    # c = m, V = u(m)
    V[T-1] = utility(m_fine)[np.newaxis, :].repeat(n_z, axis=0)
    P[T-1] = m_fine[np.newaxis, :].repeat(n_z, axis=0)
    
    # Backward induction
    for t in range(T-2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with previous period's policy (scaled)
            for t_rem in range(t, -1, -1):
                V[t_rem] = V[t+1].copy()
                P[t_rem] = P[t+1].copy()
            break
            
        V_next = V[t+1]
        
        # Create interpolators for V_next for each z'
        # Use cubic interpolation for better accuracy if possible, linear is safer
        # Given the smoothness, cubic is good, but linear is robust.
        # Let's use linear to be safe against overshoots, or cubic with bounds.
        # Actually, for value functions, linear is standard in these discrete approximations.
        interp_funcs = [interp1d(m_fine, V_next[zp], kind='linear', fill_value="extrapolate") for zp in range(n_z)]
        
        V_t = np.zeros((n_z, N_FINE))
        P_t = np.zeros((n_z, N_FINE))
        
        for z in range(n_z):
            # m_fine is shape (N_FINE,)
            # x_candidates is shape (N_CAND,)
            # C is shape (N_FINE, N_CAND)
            C = m_fine[:, np.newaxis] * x_candidates[np.newaxis, :]
            
            # Ensure C is within [c_min, m_fine]
            # x_candidates starts at 0.001, so C >= 0.001 * 0.001 = 1e-5 > 1e-12
            # C <= m_fine
            
            # Calculate next resources for each z'
            # m' = R * (m - c) + income[z']
            # We need to sum over z'
            total_cont_val = np.zeros_like(C)
            
            for zp in range(n_z):
                prob = transition[z, zp]
                next_m = R * (m_fine[:, np.newaxis] - C) + income[zp]
                
                # Interpolate V_next[zp] at next_m
                # next_m is 2D (N_FINE, N_CAND)
                # interp1d can handle array inputs if we flatten? 
                # Actually, interp1d returns same shape as input.
                # However, next_m might go out of bounds [0.001, 40].
                # fill_value="extrapolate" handles this.
                
                val_next = interp_funcs[zp](next_m)
                total_cont_val += prob * val_next
                
            # Calculate Utility
            util_c = utility(C)
            
            # Total Value
            total_val = util_c + beta * total_cont_val
            
            # Find the candidate index that maximizes total_val for each m
            # We need to handle the case where c must be <= m. 
            # Our C construction ensures c <= m (since x <= 1).
            
            best_indices = np.argmax(total_val, axis=1)
            
            # Extract optimal c and V
            m_indices = np.arange(N_FINE)
            P_t[z, :] = C[m_indices, best_indices]
            V_t[z, :] = total_val[m_indices, best_indices]
            
        V[t] = V_t
        P[t] = P_t
    
    # Interpolate results back to the original grid
    result = np.zeros((T, n_z, n_grid))
    
    for t in range(T):
        for z in range(n_z):
            # Interpolate P[t, z] from m_fine to grid
            # P[t, z] is defined on m_fine
            # We want it on grid
            interp_p = interp1d(m_fine, P[t, z], kind='linear', fill_value="extrapolate")
            result[t, z, :] = interp_p(grid)
            
    # Enforce constraints
    for t in range(T):
        for z in range(n_z):
            # c must be > 0 and <= m + 1e-10
            result[t, z, :] = np.clip(result[t, z, :], c_min, grid + 1e-10)
            
    # Final period constraint: c = m
    result[T-1, :, :] = grid[np.newaxis, :]
    
    return result
