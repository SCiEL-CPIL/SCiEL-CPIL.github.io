"""Consumption-saving policy via backward induction with FOC-based continuous search."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def solve_consumption(model, resource_grid, budget):
    """
    Solve the consumption-saving problem via backward induction using
    first-order condition (FOC) based continuous search for optimal consumption.
    
    Parameters:
    -----------
    model : dict with keys 'beta', 'gamma', 'R', 'income', 'transition', 'horizon'
    resource_grid : np.array of shape (101,)
    budget : int (number of consumption grid points per state)
    
    Returns:
    --------
    consumption : np.array of shape (6, 2, 101)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = len(income)
    m_min = m_grid[0]
    m_max = m_grid[-1]
    
    # Precompute utility and marginal utility
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            return np.log(np.maximum(c, 1e-12))
        def marginal_utility(c):
            c_safe = np.maximum(c, 1e-12)
            return 1.0 / c_safe
    else:
        one_minus_gamma = 1.0 - gamma
        def utility(c):
            return np.power(np.maximum(c, 1e-12), one_minus_gamma) / one_minus_gamma
        def marginal_utility(c):
            c_safe = np.maximum(c, 1e-12)
            return np.power(c_safe, -gamma)
    
    # Terminal condition: at t=horizon-1, consume all resources
    consumption = np.zeros((horizon, n_z, n_m))
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    
    # Value function at terminal: V(m, z) = u(m) for all z
    V_next = np.tile(utility(m_grid)[np.newaxis, :], (n_z, 1))
    
    # Helper: compute V'_next on the grid via central differences
    def compute_V_prime(V_next_arr):
        V_prime_next = np.zeros((n_z, n_m))
        for zp in range(n_z):
            V_prime_next[zp, 0] = (V_next_arr[zp, 1] - V_next_arr[zp, 0]) / (m_grid[1] - m_grid[0])
            V_prime_next[zp, -1] = (V_next_arr[zp, -1] - V_next_arr[zp, -2]) / (m_grid[-1] - m_grid[-2])
            if n_m > 2:
                V_prime_next[zp, 1:-1] = (V_next_arr[zp, 2:] - V_next_arr[zp, :-2]) / (m_grid[2:] - m_grid[:-2])
        return V_prime_next
    
    # Precompute V'_next for terminal
    V_prime_next = compute_V_prime(V_next)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fill remaining with consume-all heuristic
            for tt in range(t, horizon - 1):
                consumption[tt] = m_grid[np.newaxis, :]
            break
        
        V_curr = np.zeros((n_z, n_m))
        C_opt = np.zeros((n_z, n_m))
        
        for z in range(n_z):
            for m_idx in range(n_m):
                m_val = m_grid[m_idx]
                
                # Bisection on the FOC: find c such that objective'(c) = 0
                # c in (0, m_val)
                c_lo = 1e-12
                c_hi = m_val
                
                # Define derivative function
                def deriv_at(c):
                    c_safe = max(c, 1e-12)
                    m_minus_c = max(m_val - c_safe, 0.0)
                    m_prime = R * m_minus_c + income  # shape (n_z,)
                    m_prime_clipped = np.clip(m_prime, m_min, m_max)
                    
                    # Interpolate V'_next at m_prime_clipped
                    V_prime_at_mp = np.zeros(n_z)
                    for zp in range(n_z):
                        V_prime_at_mp[zp] = np.interp(m_prime_clipped[zp], m_grid, V_prime_next[zp, :])
                    
                    mu_c = marginal_utility(c_safe)
                    return mu_c - beta * R * np.dot(transition[z], V_prime_at_mp)
                
                # Check boundaries
                deriv_lo = deriv_at(c_lo)
                deriv_hi = deriv_at(c_hi)
                
                if deriv_lo <= 0:
                    c_opt = c_lo
                elif deriv_hi >= 0:
                    c_opt = c_hi
                else:
                    # Bisection
                    c_lo_b = c_lo
                    c_hi_b = c_hi
                    for _ in range(40):
                        c_mid = 0.5 * (c_lo_b + c_hi_b)
                        deriv_mid = deriv_at(c_mid)
                        if deriv_mid > 0:
                            c_lo_b = c_mid
                        else:
                            c_hi_b = c_mid
                        if abs(c_hi_b - c_lo_b) < 1e-12 * m_val:
                            break
                    c_opt = 0.5 * (c_lo_b + c_hi_b)
                
                c_opt = np.clip(c_opt, 1e-12, m_val)
                C_opt[z, m_idx] = c_opt
                
                # Compute value
                c_safe = max(c_opt, 1e-12)
                m_minus_c = max(m_val - c_safe, 0.0)
                m_prime = R * m_minus_c + income
                m_prime_clipped = np.clip(m_prime, m_min, m_max)
                
                cont = np.zeros(n_z)
                for zp in range(n_z):
                    cont[zp] = np.interp(m_prime_clipped[zp], m_grid, V_next[zp, :])
                cont_val = np.dot(transition[z], cont)
                
                V_curr[z, m_idx] = utility(c_safe) + beta * cont_val
        
        consumption[t] = C_opt
        V_next = V_curr
        V_prime_next = compute_V_prime(V_next)
    
    # Ensure terminal condition: c[5] = grid
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    
    # Validate: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            for m_idx in range(n_m):
                m_val = m_grid[m_idx]
                c_val = consumption[t, z, m_idx]
                if c_val <= 0:
                    consumption[t, z, m_idx] = 1e-12
                if c_val > m_val + 1e-10:
                    consumption[t, z, m_idx] = m_val
    
    return consumption
# EVOLVE-BLOCK-END
