import numpy as np
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    T = model['horizon']
    
    n_grid = len(resource_grid)
    n_states = len(income)
    
    c_policy = np.zeros((T, n_states, n_grid))
    V = np.zeros((T, n_states, n_grid))
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
            
    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    # Terminal period
    c_policy[T-1, :, :] = resource_grid[None, None, :]
    V[T-1, :, :] = utility(resource_grid)[None, :]
    
    # Backward induction with vectorized grid search
    n_cands = 200
    fracs = np.linspace(1e-12, 1.0, n_cands)
    
    for t in range(T - 2, -1, -1):
        for z in range(n_states):
            # Candidates: shape (n_grid, n_cands)
            cands = np.outer(resource_grid, fracs)
            
            # Next wealths: shape (n_grid, n_cands, n_states)
            m_next = R * (resource_grid[:, None, None] - cands[:, :, None]) + income[None, None, :]
            
            # Interpolate V[t+1] for all next wealths
            E_V_next = np.zeros((n_grid, n_cands))
            for zp in range(n_states):
                m_flat = m_next[:, :, zp].ravel()
                v_interp_flat = np.interp(m_flat, resource_grid, V[t+1, zp, :])
                v_interp = v_interp_flat.reshape((n_grid, n_cands))
                E_V_next += transition[z, zp] * v_interp
            
            # Current utility
            U_c = utility(cands)
            
            # Total value
            Total_V = U_c + beta * E_V_next
            
            # Find optimal index
            idx = np.argmax(Total_V, axis=1)
            
            # Extract optimal c and V
            c_policy[t, z, :] = cands[np.arange(n_grid), idx]
            V[t, z, :] = Total_V[np.arange(n_grid), idx]
                
    return c_policy
