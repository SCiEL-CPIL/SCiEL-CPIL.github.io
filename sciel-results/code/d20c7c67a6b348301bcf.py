import numpy as np
import time
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    transition = np.asarray(model['transition'])
    horizon = model['horizon']
    m_grid = np.asarray(resource_grid)
    
    N_m = len(m_grid)
    N_z = len(income)
    
    # Pre-compute utility function
    def util(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
    
    # Initialize value function for the terminal period (t = horizon - 1)
    # At t = horizon - 1, c = m, so V(m, z) = u(m)
    # We will compute c for t = horizon - 2 down to 0
    # V_next will hold V_{t+1}
    V_next = np.zeros((N_m, N_z))
    for z in range(N_z):
        V_next[:, z] = util(m_grid)
        
    # Storage for policies: shape (horizon, N_z, N_m)
    policies = np.zeros((horizon, N_z, N_m))
    
    # Backward induction
    # t goes from horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > DEADLINE:
            # If time runs out, break and return what we have.
            # This is unlikely given the small problem size.
            break
            
        V_curr = np.zeros((N_m, N_z))
        c_curr = np.zeros((N_z, N_m))
        
        for z in range(N_z):
            for i_m, m in enumerate(m_grid):
                # Define objective to minimize (negative of value)
                # Value = u(c) + beta * sum_z' P[z,z'] * V_next(next_m, z')
                
                def neg_obj(c):
                    if c < 0 or c > m + 1e-10:
                        return 1e10 # Penalty
                    c_val = max(c, 1e-12)
                    u_c = util(c_val)
                    
                    next_m_vals = R * (m - c_val) + income
                    # Calculate expected continuation value
                    cont_val = 0.0
                    for zp in range(N_z):
                        p_zp = transition[z, zp]
                        if p_zp > 0:
                            nm = next_m_vals[zp]
                            # np.interp clamps to grid boundaries by default
                            v_zp = np.interp(nm, m_grid, V_next[:, zp])
                            cont_val += p_zp * v_zp
                            
                    total_val = u_c + beta * cont_val
                    return -total_val

                # Optimize c in [1e-6, m]
                # Using 'bounded' method (Brent)
                res = minimize_scalar(neg_obj, bounds=(1e-6, m), method='bounded')
                
                c_opt = res.x
                # Clamp to ensure validity
                if c_opt < 1e-6:
                    c_opt = 1e-6
                if c_opt > m:
                    c_opt = m
                    
                c_curr[z, i_m] = c_opt
                V_curr[i_m, z] = -res.fun
        
        policies[t] = c_curr
        V_next = V_curr
        
    # Fill the last period (t = horizon - 1)
    # c_{T-1} = m
    for z in range(N_z):
        policies[horizon - 1, z, :] = m_grid
        
    return policies
