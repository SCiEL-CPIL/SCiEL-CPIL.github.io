"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    n_z = len(income)
    
    # Dense internal grid for backward induction
    # 501 points provides good resolution for the policy function
    m_grid = np.geomspace(0.001, 40, 501)
    n_m = len(m_grid)
    
    # Utility functions and derivatives
    def u(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c_safe)
        else:
            return c_safe ** (1.0 - gamma) / (1.0 - gamma)
    
    def u_prime(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c_safe
        else:
            return c_safe ** (-gamma)
            
    def u_double_prime(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / (c_safe ** 2)
        else:
            return -gamma * c_safe ** (-gamma - 1)

    # Policy array: c[t, z, m_index]
    c_policy = np.zeros((horizon, n_z, n_m))
    
    # Terminal: consume all
    for z in range(n_z):
        c_policy[horizon - 1, z, :] = m_grid
    
    # Value function on m_grid: V[t, z, m_index]
    V = np.zeros((horizon, n_z, n_m))
    for z in range(n_z):
        V[horizon - 1, z, :] = u(m_grid)
    
    # Splines for Value function
    # V_splines[t] is a dict: {z: CubicSpline}
    V_splines = [None] * horizon
    
    # Initialize terminal splines
    V_splines[horizon - 1] = {}
    for z in range(n_z):
        V_splines[horizon - 1][z] = CubicSpline(m_grid, V[horizon - 1, z, :], bc_type='natural')
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            break
        
        next_splines = V_splines[t + 1]
        
        for z in range(n_z):
            for i in range(n_m):
                m_i = m_grid[i]
                
                if m_i < 1e-10:
                    c_policy[t, z, i] = m_i
                    V[t, z, i] = u(m_i)
                    continue
                
                # Define Euler Equation Residual f(c)
                # f(c) = u'(c) - beta * R * sum_z' P[z,z'] * V'_{t+1,z'}(R(m-c) + y')
                
                c_lo = 1e-12
                c_hi = m_i * (1.0 - 1e-10)
                
                # Initial guess: use policy from previous step (t+1) as a heuristic
                # or simply m_i / 2. Let's use m_i * 0.5 for stability.
                c_curr = 0.5 * m_i
                c_curr = np.clip(c_curr, c_lo, c_hi)
                
                converged = False
                max_iter = 20
                
                for _ in range(max_iter):
                    # Calculate m' for each z'
                    m_save = R * (m_i - c_curr)
                    
                    v_prime_sum = 0.0
                    v_double_prime_sum = 0.0
                    
                    for zp in range(n_z):
                        prob = transition[z, zp]
                        if prob < 1e-15:
                            continue
                        m_zp = m_save + income[zp]
                        # Clip to domain of spline
                        m_zp_clipped = np.clip(m_zp, m_grid[0], m_grid[-1])
                        
                        # Get values and derivatives from spline
                        # CubicSpline supports order=0, 1, 2
                        v_val = next_splines[zp](m_zp_clipped, 0)
                        v_prime_val = next_splines[zp](m_zp_clipped, 1)
                        v_double_prime_val = next_splines[zp](m_zp_clipped, 2)
                        
                        v_prime_sum += prob * v_prime_val
                        v_double_prime_sum += prob * v_double_prime_val
                    
                    # f(c)
                    f_val = u_prime(c_curr) - beta * R * v_prime_sum
                    
                    # f'(c)
                    # f'(c) = u''(c) - beta * R * sum P * V'' * d(m')/dc
                    # d(m')/dc = -R
                    # f'(c) = u''(c) + beta * R^2 * sum P * V''
                    f_prime_val = u_double_prime(c_curr) + beta * (R ** 2) * v_double_prime_sum
                    
                    # Check for convergence
                    if abs(f_val) < 1e-10:
                        converged = True
                        break
                    
                    # Newton step
                    if abs(f_prime_val) < 1e-15:
                        # Derivative is flat, use bisection or break
                        break
                    
                    c_new = c_curr - f_val / f_prime_val
                    
                    # If Newton step goes out of bounds, use bisection
                    if c_new <= c_lo or c_new >= c_hi:
                        # Bisection logic
                        if f_val > 0:
                            # u'(c) too high, need higher c (since u' is decreasing)
                            # Wait, f(c) = u' - target. If f>0, u' > target.
                            # u' is decreasing in c. So we need larger c to lower u'.
                            c_lo = c_curr
                        else:
                            c_hi = c_curr
                        c_new = 0.5 * (c_lo + c_hi)
                    
                    c_curr = c_new
                    
                    if abs(c_new - c_curr) < 1e-12:
                        converged = True
                        break
                
                # If not converged, keep the last estimate
                c_opt = np.clip(c_curr, c_lo, c_hi)
                
                c_policy[t, z, i] = c_opt
                
                # Compute V[t, z, i]
                m_save = R * (m_i - c_opt)
                total_v = 0.0
                for zp in range(n_z):
                    prob = transition[z, zp]
                    if prob < 1e-15:
                        continue
                    m_zp = m_save + income[zp]
                    m_zp_clipped = np.clip(m_zp, m_grid[0], m_grid[-1])
                    v_val = next_splines[zp](m_zp_clipped, 0)
                    total_v += prob * v_val
                
                V[t, z, i] = u(c_opt) + beta * total_v
        
        # Build splines for current period
        V_splines[t] = {}
        for z in range(n_z):
            V_splines[t][z] = CubicSpline(m_grid, V[t, z, :], bc_type='natural')
    
    # Interpolate to resource_grid
    result = np.zeros((horizon, n_z, len(resource_grid)))
    for t in range(horizon):
        for z in range(n_z):
            try:
                c_spline = CubicSpline(m_grid, c_policy[t, z, :], bc_type='natural')
                result[t, z, :] = c_spline(resource_grid)
            except Exception:
                result[t, z, :] = np.interp(resource_grid, m_grid, c_policy[t, z, :])
    
    # Enforce constraints
    for t in range(horizon):
        for z in range(n_z):
            result[t, z, :] = np.maximum(result[t, z, :], 1e-12)
            result[t, z, :] = np.minimum(result[t, z, :], resource_grid + 1e-10)
    
    # Terminal condition
    for z in range(n_z):
        result[horizon - 1, z, :] = resource_grid
    
    # Ensure finiteness
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=1e-12, posinf=1.0, neginf=1e-12)
        for t in range(horizon):
            for z in range(n_z):
                result[t, z, :] = np.maximum(result[t, z, :], 1e-12)
                result[t, z, :] = np.minimum(result[t, z, :], resource_grid + 1e-10)
        for z in range(n_z):
            result[horizon - 1, z, :] = resource_grid
    
    return result
# EVOLVE-BLOCK-END
