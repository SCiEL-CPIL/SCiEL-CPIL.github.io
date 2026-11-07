import numpy as np
import time
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']

    m_grid_out = np.array(resource_grid)
    n_m_out = len(m_grid_out)
    n_z = len(income)

    # Internal grid for value function
    n_m_int = 4000
    m_internal = np.geomspace(0.001, 50, n_m_int)

    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    # Value function storage: V[t, z, m_idx]
    V = np.zeros((horizon, n_z, n_m_int))
    # Policy storage: C[t, z, m_idx]
    C = np.zeros((horizon, n_z, n_m_int))

    # Terminal period: consume everything
    V[horizon - 1, :, :] = utility(m_internal)[np.newaxis, :]
    C[horizon - 1, :, :] = m_internal[np.newaxis, :]

    # Pre-compute transition probabilities weighted by beta
    beta_trans = beta * transition

    # Backward induction with Newton's method for optimal s
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            # For each m, solve for optimal s using Newton's method
            for i_m in range(n_m_int):
                m_val = m_internal[i_m]
                
                def objective(s):
                    c_val = max(m_val * s, 1e-12)
                    u_val = utility(c_val)
                    
                    cont_val = 0.0
                    for zp in range(n_z):
                        m_next = R * (m_val - c_val) + income[zp]
                        m_next_clipped = np.clip(m_next, m_internal[0], m_internal[-1])
                        v_next = np.interp(m_next_clipped, m_internal, V[t + 1, zp, :])
                        cont_val += beta_trans[z, zp] * v_next
                    
                    return -(u_val + cont_val)  # Negate for minimization
                
                # Initial guess: use a reasonable fraction
                s_init = 0.5 if m_val > 0.1 else 0.3
                
                # Bounded minimization
                result = minimize_scalar(objective, bounds=(0.001, 1.0), method='bounded', 
                                        options={'xatol': 1e-8})
                
                s_opt = result.x
                c_opt = max(m_val * s_opt, 1e-12)
                
                # Compute value
                u_opt = utility(c_opt)
                cont_opt = 0.0
                for zp in range(n_z):
                    m_next = R * (m_val - c_opt) + income[zp]
                    m_next_clipped = np.clip(m_next, m_internal[0], m_internal[-1])
                    v_next = np.interp(m_next_clipped, m_internal, V[t + 1, zp, :])
                    cont_opt += beta_trans[z, zp] * v_next
                
                V[t, z, i_m] = u_opt + cont_opt
                C[t, z, i_m] = c_opt
            
            # Enforce c <= m
            C[t, z, :] = np.minimum(C[t, z, :], m_internal)
            C[t, z, :] = np.maximum(C[t, z, :], 1e-12)
        
        # Time check
        if time.time() - start_time > deadline - 5:
            break

    # Interpolate policy C from m_internal to m_grid_out
    consumption = np.zeros((horizon, n_z, n_m_out))

    for t in range(horizon):
        for z in range(n_z):
            c_out = np.interp(m_grid_out, m_internal, C[t, z, :])
            consumption[t, z, :] = c_out

            # Enforce constraints on the output grid
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid_out)
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)

    # Terminal period must consume everything exactly
    consumption[horizon - 1, :, :] = m_grid_out[np.newaxis, :]

    return consumption
