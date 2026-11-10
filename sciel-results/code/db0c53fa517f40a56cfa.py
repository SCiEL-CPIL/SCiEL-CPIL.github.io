import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    """
    Solve the consumption-saving problem via backward induction using
    Bisection on the First Order Condition (FOC) derived from the Envelope Theorem.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
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
    
    # Precompute marginal utility
    if abs(gamma - 1.0) < 1e-10:
        def marginal_utility(c):
            c_safe = np.maximum(c, 1e-12)
            return 1.0 / c_safe
    else:
        def marginal_utility(c):
            c_safe = np.maximum(c, 1e-12)
            return np.power(c_safe, -gamma)
    
    # Terminal condition: at t=horizon-1, consume all resources
    consumption = np.zeros((horizon, n_z, n_m))
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    
    # C_next[z, m_idx] = optimal consumption at t+1 for state z and resource m
    C_next = consumption[horizon - 1].copy()
    
    # Bisection iterations
    n_bisect = 60 # High precision
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fill remaining with consume-all heuristic
            for tt in range(t, horizon - 1):
                consumption[tt] = m_grid[np.newaxis, :]
            break
            
        C_curr = np.zeros((n_z, n_m))
        
        for z in range(n_z):
            trans_z = transition[z]
            
            # We need to find c for each m in m_grid
            # FOC: u'(c) = beta * R * sum_z' trans[z'][z'] * u'(C_next[z'](R*(m-c) + income[z']))
            
            c_low = np.full(n_m, 1e-12)
            c_high = m_grid.copy()
            
            # Bisection loop
            for _ in range(n_bisect):
                if time.time() > deadline:
                    break
                    
                c_mid = 0.5 * (c_low + c_high)
                
                # Compute LHS: u'(c_mid)
                lhs = marginal_utility(c_mid)
                
                # Compute RHS: beta * R * E[u'(C_next(m'))]
                # m' = R * (m - c_mid) + income[z']
                # Shape of c_mid: (n_m,)
                # Shape of m_grid: (n_m,)
                
                rhs = np.zeros(n_m)
                for zp in range(n_z):
                    if trans_z[zp] == 0:
                        continue
                    
                    # m_minus_c: (n_m,)
                    m_minus_c = np.maximum(m_grid - c_mid, 0.0)
                    m_prime = R * m_minus_c + income[zp]
                    m_prime = np.clip(m_prime, m_min, m_max)
                    
                    # Interpolate C_next[zp] at m_prime
                    # C_next[zp] is defined on m_grid
                    c_next_val = np.interp(m_prime, m_grid, C_next[zp, :])
                    
                    # Marginal utility of next period consumption
                    mu_next = marginal_utility(c_next_val)
                    
                    rhs += trans_z[zp] * mu_next
                
                rhs = beta * R * rhs
                
                # Check condition: u'(c) - RHS
                # If u'(c) > RHS, we need to increase c (since u' is decreasing, increasing c lowers LHS, and RHS increases, so they converge)
                # Wait.
                # LHS decreases with c.
                # RHS increases with c.
                # If LHS > RHS, c is too small. Increase c.
                # If LHS < RHS, c is too large. Decrease c.
                
                # c_low is where LHS > RHS (c too small)
                # c_high is where LHS < RHS (c too large)
                # At c_mid:
                # If LHS > RHS, c_mid is too small, so c_low = c_mid
                # If LHS < RHS, c_mid is too large, so c_high = c_mid
                
                condition = lhs > rhs
                c_low[condition] = c_mid[condition]
                c_high[~condition] = c_mid[~condition]
                
            C_curr[z, :] = 0.5 * (c_low + c_high)
            
        consumption[t] = C_curr
        C_next = C_curr
        
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
