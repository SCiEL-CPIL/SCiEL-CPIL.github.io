import numpy as np
import time
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    H = model['horizon']
    
    m_grid = resource_grid
    M = len(m_grid)
    
    def utility_clean(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    V = np.zeros((H + 1, 2, M))
    C = np.zeros((H, 2, M))
    
    # Terminal condition
    V[5, :, :] = utility_clean(m_grid)[np.newaxis, np.newaxis, :]
    C[5, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    # Use a dense grid for m' to evaluate continuation values
    m_prime_min = min(income)
    m_prime_max = R * m_grid[-1] + max(income)
    
    for t in range(4, -1, -1):
        for z in range(2):
            p0 = transition[z, 0]
            p1 = transition[z, 1]
            
            def objective_neg(c, m, z, t):
                c = max(c, 1e-12)
                u_c = utility_clean(np.array([c]))[0]
                
                m_prime_0 = R * (m - c) + income[0]
                m_prime_1 = R * (m - c) + income[1]
                
                # Interpolate V[t+1, z', :] at m_prime
                v_next_0 = np.interp(m_prime_0, m_grid, V[t+1, 0, :])
                v_next_1 = np.interp(m_prime_1, m_grid, V[t+1, 1, :])
                
                exp_v = p0 * v_next_0 + p1 * v_next_1
                return -(u_c + beta * exp_v)
            
            best_val = np.zeros(M)
            best_c = np.zeros(M)
            
            for i in range(M):
                m = m_grid[i]
                # Minimize negative objective
                res = minimize_scalar(
                    objective_neg,
                    bounds=(1e-12, m),
                    args=(m, z, t),
                    method='bounded',
                    options={'xatol': 1e-10}
                )
                best_c[i] = max(res.x, 1e-12)
                best_val[i] = -res.fun
            
            V[t, z, :] = best_val
            C[t, z, :] = best_c
            
    return C
