import numpy as np

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_m = len(resource_grid)
    n_z = 2
    
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
    
    V = np.zeros((horizon, n_z, n_m))
    c_opt = np.zeros((horizon, n_z, n_m))
    
    # Last period
    for z in range(n_z):
        c_opt[horizon - 1, z, :] = resource_grid
        V[horizon - 1, z, :] = util(resource_grid)
    
    # Precompute interpolation for continuation values
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            # For each m, we need to find optimal c
            # Vectorize over m
            # c_grid shape: (n_cand, n_m)
            n_cand = 2000
            c_fractions = np.linspace(1e-12, 1.0, n_cand)  # c/m ratio
            # But c must be >= 1e-12, so for small m, c_min might be > m*1e-12
            # Actually, let's just use c values directly
            
            # For each m, c ranges from max(1e-12, eps) to m
            # Let's use a fixed number of candidates
            # c_grid[i, j] = candidate consumption for m[j]
            
            # Better: for each m, create c grid
            # This is hard to vectorize easily, so let's do it per m but vectorize the continuation
            
            for i, m in enumerate(resource_grid):
                if m < 1e-10:
                    c_opt[t, z, i] = 1e-12
                    # Continuation value at m' for each z'
                    cont = np.zeros(n_z)
                    for zp in range(n_z):
                        mp = R * (m - 1e-12) + income[zp]
                        if mp <= resource_grid[0]:
                            cont[zp] = V[t+1, zp, 0]
                        elif mp >= resource_grid[-1]:
                            cont[zp] = V[t+1, zp, -1]
                        else:
                            cont[zp] = np.interp(mp, resource_grid, V[t+1, zp, :])
                    V[t, z, i] = util(1e-12) + beta * np.dot(transition[z], cont)
                    continue
                
                # Create fine grid of c values
                c_grid = np.linspace(1e-12, m, n_cand)
                # Compute m' for each z'
                m_minus_c = m - c_grid  # shape (n_cand,)
                # For each z', m' = R * m_minus_c + income[z']
                # Shape: (n_z, n_cand)
                m_prime = np.zeros((n_z, n_cand))
                for zp in range(n_z):
                    m_prime[zp, :] = R * m_minus_c + income[zp]
                
                # Interpolate V[t+1, zp, :] at m_prime[zp, :]
                # Clip m_prime to grid range
                m_prime_clipped = np.clip(m_prime, resource_grid[0], resource_grid[-1])
                V_interp = np.zeros((n_z, n_cand))
                for zp in range(n_z):
                    V_interp[zp, :] = np.interp(m_prime_clipped[zp, :], resource_grid, V[t+1, zp, :])
                
                # Continuation value: sum_z' transition[z, zp] * V_interp[zp, :]
                cont = np.dot(transition[z], V_interp)  # shape (n_cand,)
                
                # Objective: util(c) + beta * cont
                obj = util(c_grid) + beta * cont
                best_idx = np.argmax(obj)
                c_opt[t, z, i] = c_grid[best_idx]
                V[t, z, i] = obj[best_idx]
    
    return c_opt
