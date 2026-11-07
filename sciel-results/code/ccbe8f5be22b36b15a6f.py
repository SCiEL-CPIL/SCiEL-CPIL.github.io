import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 120
    
    # Extract model parameters
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Internal grid for backward induction
    # Use a fine grid to ensure accurate interpolation
    n_internal = 1000
    m_internal = np.geomspace(0.001, 40, n_internal)
    
    # Output grid
    n_out = len(resource_grid)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
            
    def utility_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    # Initialize Value Function
    V_next = np.zeros((n_internal, 2))
    
    # Policy arrays for internal grid
    policy_internal = np.zeros((horizon, 2, n_internal))
    
    # Candidate ratios
    # High resolution to minimize discretization error
    n_candidates = 5000
    ratios = np.linspace(0.001, 1.0, n_candidates)
    
    # Backward Induction
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 10:
            break
            
        V_curr = np.zeros((n_internal, 2))
        
        for z in range(2):
            probs = transition[z]
            
            # Candidate consumption matrix: (n_internal, n_candidates)
            C = m_internal[:, np.newaxis] * ratios[np.newaxis, :]
            
            # Calculate m'
            m_next_0 = R * (m_internal[:, np.newaxis] - C) + income[0]
            m_next_1 = R * (m_internal[:, np.newaxis] - C) + income[1]
            
            # Flatten for interpolation
            m_next_0_flat = m_next_0.flatten()
            m_next_1_flat = m_next_1.flatten()
            
            # Interpolate V_next using PCHIP
            # PCHIP is better for economic value functions (monotonic, smooth)
            # We need to handle the case where m_next is outside [m_internal[0], m_internal[-1]]
            # PCHIP extrapolates linearly by default? Or we can use fill_value='extrapolate'
            # Actually, m_next_min is ~0.5. m_internal starts at 0.001.
            # m_next_max is ~42. m_internal ends at 40.
            # We should extend the grid or handle extrapolation.
            # Let's extend m_internal slightly to cover the range.
            # Or just use np.interp which clamps. But PCHIP is better.
            # Let's create a slightly larger grid for interpolation if needed, 
            # or rely on the fact that m_next is mostly within range.
            # For m_next > 40, linear extrapolation is reasonable.
            
            # To be safe and fast, we can use PCHIP on the internal grid.
            # If m_next is out of bounds, PCHIP with fill_value='extrapolate' works.
            
            try:
                interp_V0 = PchipInterpolator(m_internal, V_next[:, 0], fill_value="extrapolate")
                interp_V1 = PchipInterpolator(m_internal, V_next[:, 1], fill_value="extrapolate")
                
                V_cont_0_flat = interp_V0(m_next_0_flat)
                V_cont_1_flat = interp_V1(m_next_1_flat)
            except Exception:
                # Fallback to linear interpolation if PCHIP fails
                V_cont_0_flat = np.interp(m_next_0_flat, m_internal, V_next[:, 0])
                V_cont_1_flat = np.interp(m_next_1_flat, m_internal, V_next[:, 1])
                
            V_cont_0 = V_cont_0_flat.reshape(n_internal, n_candidates)
            V_cont_1 = V_cont_1_flat.reshape(n_internal, n_candidates)
            
            E_V_cont = probs[0] * V_cont_0 + probs[1] * V_cont_1
            
            U_c = utility(C)
            
            Total_Value = U_c + beta * E_V_cont
            
            idx_max = np.argmax(Total_Value, axis=1)
            
            rows = np.arange(n_internal)
            V_curr[:, z] = Total_Value[rows, idx_max]
            policy_internal[t, z, :] = C[rows, idx_max]
            
        V_next = V_curr
        
    # Interpolate policy to output grid
    consumption_out = np.zeros((horizon, 2, n_out))
    for t in range(horizon):
        for z in range(2):
            consumption_out[t, z, :] = np.interp(resource_grid, m_internal, policy_internal[t, z, :])
            
    return consumption_out
