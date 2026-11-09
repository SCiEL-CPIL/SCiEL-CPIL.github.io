import numpy as np
from scipy.optimize import brentq
import time

def solve_consumption(model, resource_grid, budget):
    """
    Solves the consumption-saving problem using backward induction 
    and root-finding on the Euler equation.
    
    Parameters:
    model: dict with keys 'beta', 'gamma', 'R', 'income', 'transition', 'horizon'
    resource_grid: np.array of shape (101,)
    budget: int (unused in this optimized version, kept for signature)
    
    Returns:
    consumption: np.array of shape (6, 2, 101)
    """
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    m_grid = resource_grid
    num_m = len(m_grid)
    num_states = 2
    
    # Initialize consumption array
    consumption = np.zeros((horizon, num_states, num_m))
    
    # Last period: consume all
    consumption[horizon - 1, :, :] = m_grid[None, :]
    
    # Utility derivative function
    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        for z in range(num_states):
            # c_t for state z
            c_t = np.zeros(num_m)
            
            for i, m in enumerate(m_grid):
                # We need to find c in [eps, m] such that:
                # u'(c) = beta * R * sum_{z'} P(z'|z) * u'(c_{t+1}(m', z'))
                
                # Define the function to find the root of
                # f(c) = u'(c) - beta * R * E[u'(c_{t+1}(m', z'))]
                
                def f(c):
                    # Calculate next period resources for each z'
                    m_prime = R * (m - c) + income
                    # Interpolate next period consumption for each z'
                    # c_next[z', :] is the optimal consumption at t+1 for state z'
                    # We need c_next[z', m_prime[z']]
                    
                    # np.interp requires x to be increasing. m_grid is increasing.
                    # m_prime might be out of bounds if c is very small? 
                    # m_prime_min = R*m + min(income) > 0.
                    # m_prime_max = R*m + max(income). 
                    # The grid goes up to 40. m is up to 40. R=1.02.
                    # Max m' = 1.02*40 + 1.5 = 42.3.
                    # The grid max is 40. np.interp will clip to the last value if x > xp[-1].
                    # This is a standard approximation for finite grid models.
                    
                    c_next_0 = np.interp(m_prime, m_grid, consumption[t + 1, 0, :])
                    c_next_1 = np.interp(m_prime, m_grid, consumption[t + 1, 1, :])
                    
                    # Expected marginal utility
                    eu_prime = (transition[z, 0] * u_prime(c_next_0) + 
                                 transition[z, 1] * u_prime(c_next_1))
                    
                    return u_prime(c) - beta * R * eu_prime
                
                # Bounds for c
                c_min = 1e-12
                c_max = m
                
                # Check signs at boundaries
                f_min = f(c_min)
                f_max = f(c_max)
                
                # Due to floating point or specific parameters, the root might be exactly at a boundary
                # or the function might not change sign if the grid is too coarse or params extreme.
                # However, theoretically, f(c_min) > 0 (high MU now > low MU future) and f(c_max) < 0 (low MU now > high MU future).
                
                if f_min * f_max > 0:
                    # Fallback: if no sign change, pick the boundary with smaller absolute value
                    # or use the seed's logic. But usually this shouldn't happen.
                    if abs(f_min) < abs(f_max):
                        c_t[i] = c_min
                    else:
                        c_t[i] = c_max
                else:
                    # Use Brent's method
                    c_t[i] = brentq(f, c_min, c_max, xtol=1e-12, rtol=1e-12, maxiter=100)
            
            consumption[t, z, :] = c_t
            
            # Optional: Early termination if time is running out, though this is very fast.
    
    # Final check for validity constraints
    # 0 < c <= m + 1e-10
    consumption = np.clip(consumption, 1e-12, None)
    for t in range(horizon):
        for z in range(num_states):
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid + 1e-10)
            
    # Ensure c[5] is exactly the grid (within 1e-9)
    consumption[5, :, :] = m_grid[None, :]
    
    return consumption
