import numpy as np
from scipy.interpolate import interp1d

def solve_consumption(model, resource_grid, budget):
        beta = model['beta']
        gamma = model['gamma']
        R = model['R']
        income = model['income']
        transition = model['transition']
        horizon = model['horizon']
        
        # Internal fine grid
        N_fine = 4000
        # Extend grid to cover max possible m' (approx 42.3)
        m_fine = np.geomspace(0.001, 45, N_fine)
        
        # Precompute utility
        def U(c):
            c = np.maximum(c, 1e-12)
            if gamma == 1:
                return np.log(c)
            else:
                return c ** (1 - gamma) / (1 - gamma)

        V = np.zeros((horizon, 2, N_fine))
        c_fine = np.zeros((horizon, 2, N_fine))
        
        # Terminal condition
        V[horizon-1, :, :] = U(m_fine)
        c_fine[horizon-1, :, :] = m_fine
        
        # Candidates for c
        N_c = 2000
        c_rel = np.linspace(0.001, 1.0, N_c)
        
        # Precompute c_grid and m_primes
        # c_grid shape (N_fine, N_c)
        c_grid = m_fine[:, None] * c_rel[None, :]
        
        m_prime_0 = R * (m_fine[:, None] - c_grid) + income[0]
        m_prime_1 = R * (m_fine[:, None] - c_grid) + income[1]
        
        # Flatten for interp1d if needed? interp1d accepts 2D x if y is 1D?
        # No, interp1d expects x to be 1D.
        # We need to flatten m_prime_0, interpolate, reshape.
        # Or use a loop? No, flatten is fast.
        
        m_prime_0_flat = m_prime_0.ravel()
        m_prime_1_flat = m_prime_1.ravel()
        
        for t in range(horizon - 2, -1, -1):
            # Prepare interpolators for V[t+1]
            # We need to interpolate V[t+1, 0] and V[t+1, 1] at points m_prime_0 and m_prime_1
            
            # Using interp1d is cleaner but creating it has overhead.
            # np.interp is faster for 1D.
            # Let's stick to np.interp but flatten.
            
            V_next_0 = np.interp(m_prime_0_flat, m_fine, V[t+1, 0, :]).reshape(N_fine, N_c)
            V_next_1 = np.interp(m_prime_1_flat, m_fine, V[t+1, 1, :]).reshape(N_fine, N_c)
            
            for z in range(2):
                # Continuation value
                cont_val = beta * (transition[z, 0] * V_next_0 + transition[z, 1] * V_next_1)
                
                # Total value
                total_val = U(c_grid) + cont_val
                
                # Find argmax
                idx_max = np.argmax(total_val, axis=1)
                
                c_opt = c_grid[np.arange(N_fine), idx_max]
                V[t, z, :] = total_val[np.arange(N_fine), idx_max]
                c_fine[t, z, :] = c_opt

        # Interpolate c_fine to resource_grid
        c_out = np.zeros((horizon, 2, len(resource_grid)))
        for t in range(horizon):
            for z in range(2):
                c_out[t, z, :] = np.interp(resource_grid, m_fine, c_fine[t, z, :])
                
        # Enforce terminal condition exactly
        c_out[horizon-1, :, :] = resource_grid
        
        # Ensure validity: 0 < c <= m
        for t in range(horizon):
            for z in range(2):
                c_out[t, z, :] = np.clip(c_out[t, z, :], 1e-12, resource_grid + 1e-10)
                
        return c_out
