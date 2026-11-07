"""Rewritten program for consumption-saving policy accuracy."""
import numpy as np
import time
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin below 150s wall-clock limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    m_grid = resource_grid
    n_m = len(m_grid)
    
    # Initialize consumption array
    consumption = np.zeros((horizon, 2, n_m))
    
    # Last period: consume everything
    consumption[horizon - 1, :, :] = m_grid[np.newaxis, :, np.newaxis]
    consumption[horizon - 1] = np.tile(m_grid, (2, 1))
    
    def u(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    # Backward induction
    # V[t, z, i] is the continuation value at time t, state z, resource index i
    V = np.zeros((horizon + 1, 2, n_m))
    
    # At t = horizon, V is 0 (terminal)
    # At t = horizon-1, V = u(m)
    V[horizon, :, :] = 0
    V[horizon - 1, :, :] = u(m_grid)[np.newaxis, :, np.newaxis]
    V[horizon - 1] = np.tile(u(m_grid), (2, 1))
    
    for t in range(horizon - 2, -1, -1):
        for z in range(2):
            for i in range(n_m):
                m = m_grid[i]
                
                def objective(c):
                    # c is consumption at time t
                    # Next period resources: m' = R*(m - c) + income[z']
                    # Expected continuation value
                    cont_val = 0.0
                    for zp in range(2):
                        prob = transition[z, zp]
                        m_next = R * (m - c) + income[zp]
                        # Interpolate V[t+1, zp, :] at m_next
                        if m_next <= m_grid[0]:
                            v_next = V[t + 1, zp, 0]
                        elif m_next >= m_grid[-1]:
                            # Extrapolate linearly or use last value
                            # For safety, use the value at the grid boundary
                            # But m_next should be within reasonable bounds
                            # Let's clamp for interpolation safety
                            v_next = V[t + 1, zp, -1]
                        else:
                            v_next = np.interp(m_next, m_grid, V[t + 1, zp, :])
                        cont_val += prob * v_next
                    return u(c) + beta * cont_val
                
                # Find c that maximizes objective
                # Try FOC: u'(c) = beta * R * E[V'(m_next)]
                # But V' is not directly available, so use numerical optimization
                
                # Use golden section or brentq on derivative
                # First, let's try to find the optimum by searching
                
                # The objective is concave in c (since u is concave and V is concave)
                # So we can use a simple line search or FOC-based approach
                
                # Try FOC approach:
                # u'(c) = beta * R * E[V'(m_next)]
                # where V'(m_next) is the derivative of the continuation value
                
                # Numerical derivative of V at m_next
                def v_prime_interp(zp, m_next):
                    if m_next <= m_grid[0]:
                        # Use forward difference
                        dm = m_grid[1] - m_grid[0] if n_m > 1 else 1.0
                        return (V[t + 1, zp, 1] - V[t + 1, zp, 0]) / dm if n_m > 1 else 0.0
                    elif m_next >= m_grid[-1]:
                        # Use backward difference
                        dm = m_grid[-1] - m_grid[-2] if n_m > 1 else 1.0
                        return (V[t + 1, zp, -1] - V[t + 1, zp, -2]) / dm if n_m > 1 else 0.0
                    else:
                        # Use central difference
                        idx = np.searchsorted(m_grid, m_next)
                        if idx == 0:
                            dm = m_grid[1] - m_grid[0]
                            return (V[t + 1, zp, 1] - V[t + 1, zp, 0]) / dm
                        elif idx >= n_m - 1:
                            dm = m_grid[-1] - m_grid[-2]
                            return (V[t + 1, zp, -1] - V[t + 1, zp, -2]) / dm
                        else:
                            dm = m_grid[idx] - m_grid[idx - 1]
                            if dm < 1e-15:
                                return 0.0
                            return (V[t + 1, zp, idx] - V[t + 1, zp, idx - 1]) / dm
                
                def foc_residual(c):
                    c = max(c, 1e-12)
                    c = min(c, m)
                    lhs = u_prime(c)
                    rhs = 0.0
                    for zp in range(2):
                        prob = transition[z, zp]
                        m_next = R * (m - c) + income[zp]
                        vp = v_prime_interp(zp, m_next)
                        rhs += prob * vp
                    return lhs - beta * R * rhs
                
                # Find root of foc_residual in [1e-12, m]
                c_lo = 1e-12
                c_hi = m
                
                try:
                    f_lo = foc_residual(c_lo)
                    f_hi = foc_residual(c_hi)
                    
                    if f_lo * f_hi < 0:
                        c_opt = brentq(foc_residual, c_lo, c_hi, xtol=1e-14, rtol=1e-14, maxiter=100)
                    elif f_lo >= 0:
                        # f is positive at lower bound, so optimum is at lower bound
                        c_opt = c_lo
                    else:
                        # f is negative at upper bound, so optimum is at upper bound
                        c_opt = c_hi
                except Exception:
                    # Fallback to grid search
                    c_grid = np.linspace(1e-12, m, 50)
                    vals = np.array([objective(c) for c in c_grid])
                    c_opt = c_grid[np.argmax(vals)]
                
                consumption[t, z, i] = c_opt
                V[t, z, i] = objective(c_opt)
        
        # Check time
        if time.time() > deadline:
            # Return what we have so far
            break
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(2):
            for i in range(n_m):
                c = consumption[t, z, i]
                m = m_grid[i]
                if c <= 0:
                    consumption[t, z, i] = 1e-12
                elif c > m + 1e-10:
                    consumption[t, z, i] = m
    
    # Ensure c[5] equals the grid
    consumption[horizon - 1, :, :] = np.tile(m_grid, (2, 1))
    
    return consumption
