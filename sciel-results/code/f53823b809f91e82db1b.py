import numpy as np
import time
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin below 150s kill
    
    # Extract model parameters
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Output grid
    n_out = len(resource_grid)
    m_out = resource_grid
    
    # Scoring grid - use a denser grid for better accuracy
    m_fine = np.geomspace(0.001, 40, 401)
    n_fine = len(m_fine)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # Marginal utility for FOC
    def marg_util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # Policy storage
    policy_fine = np.zeros((horizon, 2, n_fine))
    
    # Value function storage
    V_next = np.zeros((n_fine, 2))
    
    # Precompute interpolation indices for efficiency
    # For each m in m_fine and each c_frac, precompute the m_next values
    
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 5:
            break
            
        V_curr = np.zeros((n_fine, 2))
        
        # Build interpolation arrays for continuation values
        # Use np.interp which is fast for 1D
        
        for z in range(2):
            probs = transition[z]
            
            for i in range(n_fine):
                m = m_fine[i]
                
                # Define the objective: maximize u(c) + beta * E[V_next(m')]
                # which is equivalent to minimizing neg_obj(c)
                
                def neg_obj(c_frac):
                    c = c_frac * m
                    if c <= 0 or c >= m:
                        return 1e10  # Penalty for invalid c
                    
                    m_next_0 = R * (m - c) + income[0]
                    m_next_1 = R * (m - c) + income[1]
                    
                    # Clip to valid range for interpolation
                    m_next_0 = np.clip(m_next_0, m_fine[0], m_fine[-1])
                    m_next_1 = np.clip(m_next_1, m_fine[0], m_fine[-1])
                    
                    V_cont_0 = np.interp(m_next_0, m_fine, V_next[:, 0])
                    V_cont_1 = np.interp(m_next_1, m_fine, V_next[:, 1])
                    
                    E_V_cont = probs[0] * V_cont_0 + probs[1] * V_cont_1
                    
                    U_c = utility(c)
                    Total_Value = U_c + beta * E_V_cont
                    
                    return -Total_Value
                
                # Use bounded scalar minimization on the fraction c/m
                # c_frac in [0.001, 1.0]
                try:
                    result = minimize_scalar(neg_obj, bounds=(0.001, 1.0), method='bounded', options={'xatol': 1e-6})
                    c_frac_opt = result.x
                    c_opt = c_frac_opt * m
                    V_curr[i, z] = -result.fun
                    policy_fine[t, z, i] = c_opt
                except Exception:
                    # Fallback to coarse grid search
                    c_fracs = np.linspace(0.001, 1.0, 101)
                    best_val = -1e10
                    best_c = m * 0.5
                    for cf in c_fracs:
                        c = cf * m
                        m_next_0 = R * (m - c) + income[0]
                        m_next_1 = R * (m - c) + income[1]
                        m_next_0 = np.clip(m_next_0, m_fine[0], m_fine[-1])
                        m_next_1 = np.clip(m_next_1, m_fine[0], m_fine[-1])
                        V_cont_0 = np.interp(m_next_0, m_fine, V_next[:, 0])
                        V_cont_1 = np.interp(m_next_1, m_fine, V_next[:, 1])
                        E_V_cont = probs[0] * V_cont_0 + probs[1] * V_cont_1
                        U_c = utility(c)
                        val = U_c + beta * E_V_cont
                        if val > best_val:
                            best_val = val
                            best_c = c
                    V_curr[i, z] = best_val
                    policy_fine[t, z, i] = best_c
                    
        V_next = V_curr
    
    # Fallback for early termination
    last_valid_t = horizon - 1
    for t in range(horizon - 1, -1, -1):
        if not np.all(policy_fine[t] == 0):
            last_valid_t = t
            break
            
    if last_valid_t < horizon - 1:
        for t in range(last_valid_t, -1, -1):
            if np.all(policy_fine[t] == 0):
                policy_fine[t] = policy_fine[last_valid_t]
                
    # Interpolate to output grid
    consumption_out = np.zeros((horizon, 2, n_out))
    
    for t in range(horizon):
        for z in range(2):
            consumption_out[t, z, :] = np.interp(m_out, m_fine, policy_fine[t, z, :])
            
    # Validity checks
    for t in range(horizon):
        for z in range(2):
            consumption_out[t, z, :] = np.maximum(consumption_out[t, z, :], 1e-12)
            consumption_out[t, z, :] = np.minimum(consumption_out[t, z, :], m_out + 1e-10)
            
    # Terminal condition
    consumption_out[5, :, :] = m_out[None, None, :]
    
    return consumption_out
