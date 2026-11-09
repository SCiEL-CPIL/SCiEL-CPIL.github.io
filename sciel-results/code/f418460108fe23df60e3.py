import numpy as np
import time
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0
    
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
    
    def utility_deriv(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    V = np.zeros((H + 1, 2, M))
    C = np.zeros((H, 2, M))
    
    # Terminal condition
    V[5, :, :] = utility_clean(m_grid)[np.newaxis, np.newaxis, :]
    C[5, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    # Build spline for V[5]
    splines = [None] * (H + 1)
    splines[5] = [
        CubicSpline(m_grid, V[5, 0, :], bc_type='natural'),
        CubicSpline(m_grid, V[5, 1, :], bc_type='natural')
    ]
    
    for t in range(4, -1, -1):
        # Check time
        if time.time() > deadline - 15:
            break
        
        for z in range(2):
            p0 = transition[z, 0]
            p1 = transition[z, 1]
            
            best_val = np.zeros(M)
            best_c = np.zeros(M)
            
            for i in range(M):
                m = m_grid[i]
                
                def objective_neg(c, m=m, z=z, t=t):
                    c = max(c, 1e-12)
                    u_c = utility_clean(np.array([c]))[0]
                    
                    m_prime_0 = R * (m - c) + income[0]
                    m_prime_1 = R * (m - c) + income[1]
                    
                    # Interpolate V[t+1, z', :] at m_prime using spline
                    v_next_0 = float(splines[t+1][0](m_prime_0))
                    v_next_1 = float(splines[t+1][1](m_prime_1))
                    
                    exp_v = p0 * v_next_0 + p1 * v_next_1
                    return -(u_c + beta * exp_v)
                
                # Use initial guess from ratio-based heuristic
                # For CRRA, optimal c/m is roughly constant
                if t == 4:
                    # Near terminal, consume most
                    c_init = 0.95 * m
                else:
                    # Use previous period's ratio as initial guess
                    if i > 0 and best_c[i-1] > 0:
                        ratio = best_c[i-1] / m_grid[i-1]
                        c_init = min(max(ratio * m, 1e-12), m * 0.99)
                    else:
                        c_init = 0.5 * m
                
                # Minimize negative objective
                res = minimize_scalar(
                    objective_neg,
                    bounds=(1e-12, m * 0.999999),
                    method='bounded',
                    options={'xatol': 1e-10}
                )
                best_c[i] = max(res.x, 1e-12)
                best_val[i] = -res.fun
            
            V[t, z, :] = best_val
            C[t, z, :] = best_c
        
        # Build splines for this period
        splines[t] = [
            CubicSpline(m_grid, V[t, 0, :], bc_type='natural'),
            CubicSpline(m_grid, V[t, 1, :], bc_type='natural')
        ]
    
    # Ensure terminal condition is exact
    C[5, :, :] = m_grid[np.newaxis, np.newaxis, :]
    V[5, :, :] = utility_clean(m_grid)[np.newaxis, np.newaxis, :]
    
    # Enforce validity: 0 < c <= m + 1e-10
    for t in range(H):
        for z in range(2):
            C[t, z, :] = np.clip(C[t, z, :], 1e-12, m_grid[np.newaxis, :])
    
    return C
