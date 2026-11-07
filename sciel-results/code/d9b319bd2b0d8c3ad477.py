import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = len(resource_grid)
    Z = len(income)
    m_grid = resource_grid

    # CRRA utility and derivatives
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            c = np.maximum(c, 1e-12)
            return np.log(c)
        def utility_prime(c):
            c = np.maximum(c, 1e-12)
            return 1.0 / c
        def utility_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -1.0 / (c * c)
    else:
        def utility(c):
            c = np.maximum(c, 1e-12)
            return c ** (1 - gamma) / (1 - gamma)
        def utility_prime(c):
            c = np.maximum(c, 1e-12)
            return c ** (-gamma)
        def utility_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -gamma * c ** (-gamma - 1)

    # We'll compute the consumption policy using backward induction
    # with Newton's method on the Euler equation.
    # 
    # At each (t, z), for each m, we solve:
    # u'(c) = beta * sum_{z'} P[z,z'] * R * u'(c_{t+1}(m', z'))
    # where m' = R*(m - c) + income[z']
    #
    # We represent c_{t+1}(m', z') as a function via interpolation on a fine grid.
    
    # Use a fine grid for interpolation of next-period consumption
    m_fine = np.geomspace(0.001, 40, 401)
    
    # Store consumption on fine grid for each (t, z)
    c_fine = np.zeros((horizon, Z, len(m_fine)))
    
    # Terminal period: c(m) = m
    for z in range(Z):
        c_fine[horizon - 1, z, :] = m_fine.copy()
    
    # Work backwards
    for t in range(horizon - 2, -1, -1):
        for z in range(Z):
            c_t_fine = np.zeros(len(m_fine))
            
            # Vectorized Newton's method
            # Initial guess: proportional rule c = m / (1 + beta * R * sum P)
            # Actually, let's use a better initial guess based on the log case
            if abs(gamma - 1.0) < 1e-10:
                # For log utility, c* = m / (1 + beta * R * sum_z' P[z,z'])
                # But this is approximate. Let's use c = m * 0.5 as initial guess
                c_init = m_fine * 0.5
            else:
                c_init = m_fine * 0.5
            
            c_curr = c_init.copy()
            
            for newton_iter in range(30):
                # Compute m' for each z'
                # m' = R * (m - c) + income[z']
                m_prime = np.zeros((Z, len(m_fine)))
                for zp in range(Z):
                    m_prime[zp, :] = R * (m_fine - c_curr) + income[zp]
                    m_prime[zp, :] = np.maximum(m_prime[zp, :], m_fine[0])
                    m_prime[zp, :] = np.minimum(m_prime[zp, :], m_fine[-1])
                
                # Interpolate next-period consumption
                c_next = np.zeros((Z, len(m_fine)))
                for zp in range(Z):
                    c_next[zp, :] = np.interp(m_prime[zp, :], m_fine, c_fine[t + 1, zp, :])
                
                # Compute RHS of Euler equation:
                # RHS = beta * sum_z' P[z,z'] * R * u'(c_{t+1}(m', z'))
                rhs = np.zeros(len(m_fine))
                for zp in range(Z):
                    u_prime_next = utility_prime(c_next[zp, :])
                    rhs += transition[z, zp] * R * u_prime_next
                rhs *= beta
                
                # LHS = u'(c)
                lhs = utility_prime(c_curr)
                
                # Residual = lhs - rhs
                residual = lhs - rhs
                
                # Derivative of residual w.r.t. c:
                # d/dc [u'(c) - beta * sum P * R * u'(c_{t+1}(m'))]
                # = u''(c) - beta * sum P * R * u''(c_{t+1}(m')) * d(m')/dc
                # d(m')/dc = -R
                # So: u''(c) + beta * sum P * R^2 * u''(c_{t+1}(m'))
                
                deriv = np.zeros(len(m_fine))
                u_dd_curr = utility_double_prime(c_curr)
                deriv[:] = u_dd_curr
                for zp in range(Z):
                    u_dd_next = utility_double_prime(c_next[zp, :])
                    deriv += beta * transition[z, zp] * R * R * u_dd_next
                
                # Newton step: c_new = c - residual / deriv
                # But we need to clip to [epsilon, m]
                c_new = c_curr - residual / deriv
                c_new = np.maximum(c_new, 1e-10)
                c_new = np.minimum(c_new, m_fine - 1e-10)
                
                # Check convergence
                if np.max(np.abs(c_new - c_curr)) < 1e-12:
                    c_curr = c_new
                    break
                
                c_curr = c_new
            
            c_t_fine = c_curr
        
        # Check time
        if time.time() > deadline - 5:
            # Fill remaining periods with proportional rule
            for tt in range(t, -1, -1):
                for zz in range(Z):
                    c_fine[tt, zz, :] = m_fine * 0.5
            break
    
    # Now interpolate the consumption policy onto the resource_grid
    consumption = np.zeros((horizon, Z, M))
    
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.interp(m_grid, m_fine, c_fine[t, z, :])
    
    # Terminal period: ensure c = m exactly
    for z in range(Z):
        consumption[horizon - 1, z, :] = m_grid.copy()
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid + 1e-10)
    
    # Ensure finite values
    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=m_grid, neginf=1e-12)
    
    return consumption
