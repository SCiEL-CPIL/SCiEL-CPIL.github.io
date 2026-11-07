import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = len(resource_grid)
    Z = len(income)
    m_grid = resource_grid

    # CRRA utility
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            c = np.maximum(c, 1e-12)
            return np.log(c)
    else:
        def utility(c):
            c = np.maximum(c, 1e-12)
            return c**(1 - gamma) / (1 - gamma)

    # Use a fine grid for the value function
    n_fine = 601
    m_fine = np.geomspace(0.001, 40, n_fine)

    # V[t][z] will be a 1D array of length n_fine
    V = np.zeros((horizon, Z, n_fine))

    # Terminal: V[horizon-1][z][i] = u(m_fine[i])
    for z in range(Z):
        V[horizon-1, z, :] = utility(m_fine)

    # Number of consumption grid points for backward induction
    n_c = 200
    c_fracs = np.linspace(0.001, 1.0, n_c)

    # Work backwards from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        for z in range(Z):
            # m_fine has shape (n_fine,), c_fracs has shape (n_c,)
            # We want to compute for each m in m_fine, the best c
            # m_vals: (n_fine, 1), c_fracs: (1, n_c) -> c_vals: (n_fine, n_c)
            m_vals = m_fine[:, np.newaxis]  # (n_fine, 1)
            c_vals = m_vals * c_fracs[np.newaxis, :]  # (n_fine, n_c)
            c_vals = np.maximum(c_vals, 1e-12)
            
            # Compute continuation value for all (m, c) pairs
            # m_prime = R*(m - c) + income[zp], shape (n_fine, n_c)
            cont_val = np.zeros((n_fine, n_c))
            for zp in range(Z):
                m_prime = R * (m_vals - c_vals) + income[zp]
                m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                # For each row (m_fine index), interpolate V[t+1, zp, :]
                # We need to interpolate each row of m_prime against m_fine
                # m_prime: (n_fine, n_c), for each row i, interp m_prime[i,:] using m_fine and V[t+1,zp,:]
                # Vectorized: use np.interp on each row
                v_interp = np.apply_along_axis(lambda row: np.interp(row, m_fine, V[t+1, zp, :]), 1, m_prime)
                cont_val += transition[z, zp] * v_interp
            
            obj = utility(c_vals) + beta * cont_val  # (n_fine, n_c)
            V[t, z, :] = obj.max(axis=1)
        
        # Check time
        if time.time() > deadline - 20:
            break

    # Now extract the consumption policy
    consumption = np.zeros((horizon, Z, M))
    
    # Terminal period: consume everything
    for z in range(Z):
        consumption[horizon-1, z, :] = m_grid.copy()
    
    # For t = 0 to horizon-2, use grid search with local refinement
    # Process in batches to stay within time
    for t in range(horizon - 1):
        for z in range(Z):
            # m_grid has shape (M,)
            m_vals = m_grid[:, np.newaxis]  # (M, 1)
            c_vals = m_vals * c_fracs[np.newaxis, :]  # (M, n_c)
            c_vals = np.maximum(c_vals, 1e-12)
            
            # Compute continuation value
            cont_val = np.zeros((M, n_c))
            for zp in range(Z):
                m_prime = R * (m_vals - c_vals) + income[zp]
                m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                v_interp = np.apply_along_axis(lambda row: np.interp(row, m_fine, V[t+1, zp, :]), 1, m_prime)
                cont_val += transition[z, zp] * v_interp
            
            obj = utility(c_vals) + beta * cont_val  # (M, n_c)
            best_idx = obj.argmax(axis=1)  # (M,)
            
            # Local refinement
            c_coarse = c_vals[np.arange(M), best_idx]  # (M,)
            
            # Refine around the coarse solution
            c_low = np.maximum(1e-12, c_coarse * 0.8)
            c_high = np.minimum(m_grid, c_coarse * 1.2)
            
            n_c_local = 50
            # Create local grids: (M, n_c_local)
            local_fracs = np.linspace(0, 1, n_c_local)  # (n_c_local,)
            c_local = c_low[:, np.newaxis] + local_fracs[np.newaxis, :] * (c_high - c_low)[:, np.newaxis]
            c_local = np.clip(c_local, 1e-12, m_grid[:, np.newaxis])
            
            cont_val_local = np.zeros((M, n_c_local))
            for zp in range(Z):
                m_prime_local = R * (m_vals - c_local) + income[zp]
                m_prime_local = np.clip(m_prime_local, m_fine[0], m_fine[-1])
                v_interp_local = np.apply_along_axis(lambda row: np.interp(row, m_fine, V[t+1, zp, :]), 1, m_prime_local)
                cont_val_local += transition[z, zp] * v_interp_local
            
            obj_local = utility(c_local) + beta * cont_val_local
            best_idx_local = obj_local.argmax(axis=1)
            consumption[t, z, :] = c_local[np.arange(M), best_idx_local]
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid + 1e-10)
    
    # Ensure finite values
    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=m_grid, neginf=1e-12)
    
    return consumption
