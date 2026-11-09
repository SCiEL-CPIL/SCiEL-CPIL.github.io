import numpy as np
import time
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
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
    
    # Terminal condition: t=H-1=5, c=m
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
                
                def objective_neg_f(f, m=m, z=z, t=t):
                    # f is the ratio c/m, so c = f * m
                    c = f * m
                    if c <= 1e-12:
                        return 1e10  # Large penalty
                    if c > m + 1e-10:
                        return 1e10  # Large penalty
                    
                    c_clipped = max(c, 1e-12)
                    u_c = utility_clean(np.array([c_clipped]))[0]
                    
                    m_prime_0 = R * (m - c_clipped) + income[0]
                    m_prime_1 = R * (m - c_clipped) + income[1]
                    
                    # Interpolate V[t+1, z', :] at m_prime using spline
                    # Clip m_prime to a reasonable range to avoid extreme extrapolation
                    m_prime_0_clipped = np.clip(m_prime_0, 1e-6, 100.0)
                    m_prime_1_clipped = np.clip(m_prime_1, 1e-6, 100.0)
                    
                    v_next_0 = float(splines[t+1][0](m_prime_0_clipped))
                    v_next_1 = float(splines[t+1][1](m_prime_1_clipped))
                    
                    exp_v = p0 * v_next_0 + p1 * v_next_1
                    return -(u_c + beta * exp_v)
                
                def objective_deriv_f(f, m=m, z=z, t=t):
                    # Derivative of the negative objective with respect to f
                    c = f * m
                    if c <= 1e-12:
                        return 1e10
                    if c > m + 1e-10:
                        return -1e10
                    
                    c_clipped = max(c, 1e-12)
                    u_c_deriv = utility_deriv(np.array([c_clipped]))[0] * m
                    
                    m_prime_0 = R * (m - c_clipped) + income[0]
                    m_prime_1 = R * (m - c_clipped) + income[1]
                    
                    m_prime_0_clipped = np.clip(m_prime_0, 1e-6, 100.0)
                    m_prime_1_clipped = np.clip(m_prime_1, 1e-6, 100.0)
                    
                    # Derivative of spline values
                    v_next_0_deriv = float(splines[t+1][0](m_prime_0_clipped, 1))
                    v_next_1_deriv = float(splines[t+1][1](m_prime_1_clipped, 1))
                    
                    # Chain rule: dV/dm * dm/dc * dc/df
                    # dm/dc = -R, dc/df = m
                    exp_v_deriv = (p0 * v_next_0_deriv + p1 * v_next_1_deriv) * (-R) * m
                    beta_exp_v_deriv = beta * exp_v_deriv
                    
                    return -(u_c_deriv + beta_exp_v_deriv)
                
                # Use initial guess from ratio-based heuristic
                # For CRRA, optimal c/m is roughly constant
                if t == 4:
                    # Near terminal, consume most
                    f_init = 0.95
                else:
                    # Use previous period's ratio as initial guess
                    if i > 0 and best_c[i-1] > 0:
                        ratio = best_c[i-1] / m_grid[i-1]
                        f_init = min(max(ratio, 0.01), 0.99)
                    else:
                        f_init = 0.5
                
                # Minimize negative objective over f in [0.001, 0.999999]
                # First do a coarse grid search to find a good starting point
                f_grid = np.linspace(0.01, 0.99, 30)
                obj_vals = np.array([objective_neg_f(f) for f in f_grid])
                f_coarse = f_grid[np.argmin(obj_vals)]
                
                # Then use bounded optimization with tighter tolerance
                res = minimize_scalar(
                    objective_neg_f,
                    bounds=(max(0.001, f_coarse - 0.1), min(0.999999, f_coarse + 0.1)),
                    method='bounded',
                    options={'xatol': 1e-12, 'maxiter': 100}
                )
                f_opt = res.x
                best_c[i] = max(f_opt * m, 1e-12)
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
    
    # Final high-precision polish pass for interior points
    if time.time() < deadline - 5:
        for t in range(4, -1, -1):
            for z in range(2):
                p0 = transition[z, 0]
                p1 = transition[z, 1]
                
                for i in range(M):
                    m = m_grid[i]
                    c_current = C[t, z, i]
                    f_current = c_current / m if m > 1e-10 else 0.5
                    
                    def objective_neg_f_polish(f, m=m, z=z, t=t, c_current=c_current):
                        c = f * m
                        if c <= 1e-12:
                            return 1e10
                        if c > m + 1e-10:
                            return 1e10
                        
                        c_clipped = max(c, 1e-12)
                        u_c = utility_clean(np.array([c_clipped]))[0]
                        
                        m_prime_0 = R * (m - c_clipped) + income[0]
                        m_prime_1 = R * (m - c_clipped) + income[1]
                        
                        m_prime_0_clipped = np.clip(m_prime_0, 1e-6, 100.0)
                        m_prime_1_clipped = np.clip(m_prime_1, 1e-6, 100.0)
                        
                        v_next_0 = float(splines[t+1][0](m_prime_0_clipped))
                        v_next_1 = float(splines[t+1][1](m_prime_1_clipped))
                        
                        exp_v = p0 * v_next_0 + p1 * v_next_1
                        return -(u_c + beta * exp_v)
                    
                    # Very tight bounds around current solution
                    lower = max(0.001, f_current - 0.01)
                    upper = min(0.999999, f_current + 0.01)
                    
                    if lower < upper:
                        res = minimize_scalar(
                            objective_neg_f_polish,
                            bounds=(lower, upper),
                            method='bounded',
                            options={'xatol': 1e-14, 'maxiter': 200}
                        )
                        f_polished = res.x
                        C[t, z, i] = max(f_polished * m, 1e-12)
                    
                    if time.time() > deadline - 2:
                        break
                if time.time() > deadline - 2:
                    break
            if time.time() > deadline - 2:
                break
    
    return C
