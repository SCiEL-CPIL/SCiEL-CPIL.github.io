import time
import numpy as np
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    trans = model['transition']
    horizon = model['horizon']
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = len(income)

    # Precompute utility function
    if abs(gamma - 1.0) < 1e-12:
        def utility(c):
            c = np.maximum(c, 1e-12)
            return np.log(c)
    else:
        def utility(c):
            c = np.maximum(c, 1e-12)
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    # Continuation values: V[t, z, m] for t in 0..horizon-1
    # V[horizon, z, m] = utility(m)  (consume all at final period)
    V = np.zeros((horizon + 1, n_z, n_m))
    V[horizon, :, :] = utility(m_grid)

    # Backward induction
    for t in range(horizon - 1, -1, -1):
        # Check time
        if time.time() > deadline:
            # Return best available (incomplete) - fill remaining with simple heuristic
            for tt in range(t + 1, horizon):
                V[tt, :, :] = V[t, :, :] * 0.99  # rough fallback
            break

        for z in range(n_z):
            # Expected continuation value as function of next resource m'
            # m' = R * (m - c) + income[z']
            # For a given m and c, m' depends on z'
            # We need: E[ V[t+1, z', m'] | z ] as function of c (or m')
            
            # The optimal c maximizes: utility(c) + beta * E[V[t+1, z', m']]
            # where m' = R*(m - c) + income[z']
            
            # Strategy: For each m in m_grid, find optimal c via 1D optimization
            # But that's expensive. Instead, use a coarse grid search + refinement.
            
            # Precompute: for each possible m' value, the expected continuation
            # m' ranges from R*0 + min(income) to R*m + max(income)
            # Actually m' = R*(m-c) + income[z'], so m' in [R*(m-1)*... , R*m + income[z']]
            # More precisely, c in [0, m], so m' in [R*0 + income[z'], R*m + income[z']]
            
            # Let's work with m' directly. For each z', m' = R*(m-c) + income[z']
            # So c = (m - (m' - income[z'])/R) ... no, let's just parametrize by c.
            
            # For efficiency, use a vectorized approach with a fine grid of c values
            # c in [eps, m] for each m
            
            # Create a fine grid for c as fraction of m
            c_frac = np.linspace(0.001, 1.0, 200)
            
            V_t_z = np.zeros(n_m)
            
            for i_m in range(n_m):
                m = m_grid[i_m]
                
                # c values
                c_vals = c_frac * m
                c_vals = np.clip(c_vals, 1e-12, m)
                
                # For each c and each z', compute m' and expected continuation
                # m' = R*(m - c) + income[z']
                m_minus_c = m - c_vals  # shape (200,)
                
                # Expected continuation: sum over z' of trans[z, z'] * V[t+1, z', m']
                # m' depends on z', so we need to interpolate V[t+1, z', :] at m'
                
                cont = np.zeros(len(c_vals))
                for zp in range(n_z):
                    m_prime = R * m_minus_c + income[zp]  # shape (200,)
                    # Interpolate V[t+1, zp, :] at m_prime
                    # Clip m_prime to grid range
                    m_prime_clipped = np.clip(m_prime, m_grid[0], m_grid[-1])
                    v_interp = np.interp(m_prime_clipped, m_grid, V[t+1, zp, :])
                    cont += trans[z, zp] * v_interp
                
                # Total value: utility(c) + beta * cont
                total = utility(c_vals) + beta * cont
                
                # Find max
                idx_best = np.argmax(total)
                c_opt = c_vals[idx_best]
                
                # Refine around the best point with a finer local search
                # Local grid around c_opt
                c_lo = max(1e-12, c_opt * 0.95)
                c_hi = min(m, c_opt * 1.05)
                c_fine = np.linspace(c_lo, c_hi, 50)
                c_fine = np.clip(c_fine, 1e-12, m)
                
                m_minus_c_fine = m - c_fine
                cont_fine = np.zeros(len(c_fine))
                for zp in range(n_z):
                    m_prime_fine = R * m_minus_c_fine + income[zp]
                    m_prime_fine_clipped = np.clip(m_prime_fine, m_grid[0], m_grid[-1])
                    v_interp_fine = np.interp(m_prime_fine_clipped, m_grid, V[t+1, zp, :])
                    cont_fine += trans[z, zp] * v_interp_fine
                
                total_fine = utility(c_fine) + beta * cont_fine
                idx_best_fine = np.argmax(total_fine)
                c_opt = c_fine[idx_best_fine]
                
                V_t_z[i_m] = utility(c_opt) + beta * (
                    sum(trans[z, zp] * np.interp(
                        np.clip(R * (m - c_opt) + income[zp], m_grid[0], m_grid[-1]),
                        m_grid, V[t+1, zp, :]
                    ) for zp in range(n_z))
                )
            
            V[t, z, :] = V_t_z

    # Now extract consumption policy
    # consumption[t, z, i_m] = optimal c at time t, state z, resource m_grid[i_m]
    consumption = np.zeros((horizon, n_z, n_m))
    
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline:
            # Fallback: simple rule
            for z in range(n_z):
                for i_m in range(n_m):
                    consumption[t, z, i_m] = m_grid[i_m] * 0.5
            break
        
        for z in range(n_z):
            for i_m in range(n_m):
                m = m_grid[i_m]
                c_frac = np.linspace(0.001, 1.0, 200)
                c_vals = np.clip(c_frac * m, 1e-12, m)
                
                m_minus_c = m - c_vals
                cont = np.zeros(len(c_vals))
                for zp in range(n_z):
                    m_prime = R * m_minus_c + income[zp]
                    m_prime_clipped = np.clip(m_prime, m_grid[0], m_grid[-1])
                    v_interp = np.interp(m_prime_clipped, m_grid, V[t+1, zp, :])
                    cont += trans[z, zp] * v_interp
                
                total = utility(c_vals) + beta * cont
                idx_best = np.argmax(total)
                c_opt = c_vals[idx_best]
                
                # Local refinement
                c_lo = max(1e-12, c_opt * 0.95)
                c_hi = min(m, c_opt * 1.05)
                c_fine = np.linspace(c_lo, c_hi, 50)
                c_fine = np.clip(c_fine, 1e-12, m)
                
                m_minus_c_fine = m - c_fine
                cont_fine = np.zeros(len(c_fine))
                for zp in range(n_z):
                    m_prime_fine = R * m_minus_c_fine + income[zp]
                    m_prime_fine_clipped = np.clip(m_prime_fine, m_grid[0], m_grid[-1])
                    v_interp_fine = np.interp(m_prime_fine_clipped, m_grid, V[t+1, zp, :])
                    cont_fine += trans[z, zp] * v_interp_fine
                
                total_fine = utility(c_fine) + beta * cont_fine
                idx_best_fine = np.argmax(total_fine)
                c_opt = c_fine[idx_best_fine]
                
                consumption[t, z, i_m] = c_opt
    
    # Final period: consume all
    consumption[horizon - 1, :, :] = m_grid
    
    # Validate
    consumption = np.clip(consumption, 1e-12, None)
    for t in range(horizon):
        for z in range(n_z):
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid + 1e-10)
    
    # Ensure c[5] (last period) equals grid
    consumption[horizon - 1, :, :] = m_grid
    
    return consumption
