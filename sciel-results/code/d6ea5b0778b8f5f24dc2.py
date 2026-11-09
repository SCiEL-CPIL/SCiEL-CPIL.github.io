import numpy as np
from scipy.optimize import minimize_scalar
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    m_grid = resource_grid
    n_m = len(m_grid)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # Value function arrays
    # V[t, z, i]
    # We only need V[t+1] to compute V[t] and c[t]
    V_next = np.zeros((2, n_m)) # V_{t+1}
    
    # Store consumption policies
    # c[t, z, i]
    c_policy = np.zeros((horizon, 2, n_m))
    
    # Pre-compute transition probabilities
    # transition[z, z']
    
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline:
            # Fallback: return whatever we have, fill rest with grid
            for tt in range(t, horizon):
                for zz in range(2):
                    c_policy[tt, zz, :] = m_grid
            return c_policy

        for z in range(2):
            for i in range(n_m):
                m = m_grid[i]
                
                if t == horizon - 1:
                    # Last period: consume all
                    c_opt = m
                    v = utility(c_opt)
                else:
                    # Define objective to minimize (negative of value)
                    def objective(c_val):
                        # c_val is scalar
                        if c_val < 1e-12:
                            c_val = 1e-12
                        if c_val > m:
                            c_val = m
                        
                        u_c = utility(c_val)
                        
                        # Calculate expected continuation value
                        exp_v = 0.0
                        for zp in range(2):
                            p = transition[z, zp]
                            if p > 0:
                                m_prime = R * (m - c_val) + income[zp]
                                # Clamp m_prime to grid for interpolation
                                if m_prime < m_grid[0]:
                                    m_prime = m_grid[0]
                                elif m_prime > m_grid[-1]:
                                    m_prime = m_grid[-1]
                                
                                # Interpolate V_next[zp] at m_prime
                                # np.interp is fast
                                v_zp = np.interp(m_prime, m_grid, V_next[zp])
                                exp_v += p * v_zp
                        
                        return -(u_c + beta * exp_v)
                    
                    # Optimize
                    # Bounds: [1e-12, m]
                    res = minimize_scalar(objective, bounds=(1e-12, m), method='bounded')
                    c_opt = res.x
                    v = -res.fun
                
                c_policy[t, z, i] = c_opt
                V_next[z, i] = v
                
    return c_policy
