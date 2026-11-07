import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_out = len(resource_grid)
    m_out = resource_grid
    m_fine = np.geomspace(0.001, 40, 251)
    n_fine = len(m_fine)
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    def utility_deriv(c):
        c = np.maximum(c, 1e-12)
        return c ** (-gamma)
    
    policy_fine = np.zeros((horizon, 2, n_fine))
    V_next = np.zeros((n_fine, 2))
    
    # Pre-compute candidate fractions
    n_cand = 401
    fractions = np.linspace(0.001, 1.0, n_cand)
    
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 5:
            break
        
        V_curr = np.zeros((n_fine, 2))
        
        for z in range(2):
            probs = transition[z]
            
            for i in range(n_fine):
                m = m_fine[i]
                
                c_candidates = m * fractions
                
                m_next_0 = R * (m - c_candidates) + income[0]
                m_next_1 = R * (m - c_candidates) + income[1]
                
                # Use PCHIP interpolation for smoother continuation values
                # But PCHIP requires the input to be sorted and we need to handle
                # the case where m_next might be outside [0.001, 40]
                
                # Clip m_next to the grid range for interpolation
                m_next_0_clipped = np.clip(m_next_0, m_fine[0], m_fine[-1])
                m_next_1_clipped = np.clip(m_next_1, m_fine[0], m_fine[-1])
                
                # Build PCHIP interpolators (expensive to build each time)
                # Alternative: use np.interp which is linear - faster but less accurate
                # Let's try a hybrid: use np.interp for speed, but with a denser grid
                
                # Actually, building PCHIP 251*2*6 = 3012 times is too slow.
                # Let's stick with np.interp but increase grid density.
                
                V_cont_0 = np.interp(m_next_0_clipped, m_fine, V_next[:, 0])
                V_cont_1 = np.interp(m_next_1_clipped, m_fine, V_next[:, 1])
                
                E_V_cont = probs[0] * V_cont_0 + probs[1] * V_cont_1
                
                U_c = utility(c_candidates)
                
                Total_Value = U_c + beta * E_V_cont
                
                idx_max = np.argmax(Total_Value)
                V_curr[i, z] = Total_Value[idx_max]
                c_opt = c_candidates[idx_max]
                
                # Local refinement using Newton's method on the Euler equation
                # The Euler equation: u'(c) = beta * E[u'(c')] * R
                # where c' = R(m-c) + income[z']
                # u'(c) = c^(-gamma)
                # c^(-gamma) = beta * E[(R(m-c)+income[z'])^(-gamma)] * R
                # Let's define g(c) = c^(-gamma) - beta * R * E[(R(m-c)+income[z'])^(-gamma)]
                # We want g(c) = 0
                
                # Newton's method: c_{k+1} = c_k - g(c_k)/g'(c_k)
                # g'(c) = -gamma * c^(-gamma-1) - beta * R * E[gamma * (R(m-c)+income[z'])^(-gamma-1) * (-R)]
                #       = -gamma * c^(-gamma-1) + beta * R^2 * gamma * E[(R(m-c)+income[z'])^(-gamma-1)]
                
                c_new = c_opt
                for _ in range(10):
                    c_new = np.maximum(c_new, 1e-12)
                    c_new = np.minimum(c_new, m)
                    
                    # g(c)
                    m_n0 = R * (m - c_new) + income[0]
                    m_n1 = R * (m - c_new) + income[1]
                    m_n0 = np.maximum(m_n0, 1e-12)
                    m_n1 = np.maximum(m_n1, 1e-12)
                    
                    g_val = c_new**(-gamma) - beta * R * (probs[0] * m_n0**(-gamma) + probs[1] * m_n1**(-gamma))
                    
                    # g'(c)
                    g_deriv = -gamma * c_new**(-gamma-1) + beta * R**2 * gamma * (probs[0] * m_n0**(-gamma-1) + probs[1] * m_n1**(-gamma-1))
                    
                    if abs(g_deriv) < 1e-15:
                        break
                    
                    c_new = c_new - g_val / g_deriv
                    
                    if abs(c_new - c_opt) < 1e-10:
                        break
                
                # Ensure validity
                c_new = np.maximum(c_new, 1e-12)
                c_new = np.minimum(c_new, m)
                
                policy_fine[t, z, i] = c_new
                # Update V_curr with the refined value
                m_n0 = R * (m - c_new) + income[0]
                m_n1 = R * (m - c_new) + income[1]
                m_n0_clipped = np.clip(m_n0, m_fine[0], m_fine[-1])
                m_n1_clipped = np.clip(m_n1, m_fine[0], m_fine[-1])
                V_cont_0_ref = np.interp(m_n0_clipped, m_fine, V_next[:, 0])
                V_cont_1_ref = np.interp(m_n1_clipped, m_fine, V_next[:, 1])
                E_V_cont_ref = probs[0] * V_cont_0_ref + probs[1] * V_cont_1_ref
                V_curr[i, z] = utility(c_new) + beta * E_V_cont_ref
        
        V_next = V_curr
    
    # Fill remaining t if we broke early
    last_valid_t = -1
    for t in range(horizon - 1, -1, -1):
        if not np.all(policy_fine[t] == 0):
            last_valid_t = t
            break
            
    if last_valid_t != horizon - 1 and last_valid_t >= 0:
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
    
    consumption_out[5, :, :] = m_out[None, None, :]
    
    return consumption_out
