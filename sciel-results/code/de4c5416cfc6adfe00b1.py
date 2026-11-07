"""Economics: consumption-saving policy accuracy - analytical FOC approach."""
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin below 120s (150s kill)
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']
    
    m_grid = np.asarray(resource_grid)
    n_m = len(m_grid)
    n_z = len(income)
    
    # Consumption array: shape (horizon, n_z, n_m)
    consumption = np.zeros((horizon, n_z, n_m))
    
    # Last period: consume everything
    consumption[horizon - 1, :, :] = m_grid[np.newaxis, :, np.newaxis] * np.ones((1, n_z, 1))
    consumption[horizon - 1, :, :] = m_grid[np.newaxis, :]
    
    # For CRRA utility:
    # u(c) = log(c) if gamma == 1, else c^(1-gamma)/(1-gamma)
    # u'(c) = 1/c if gamma == 1, else c^(-gamma)
    # u''(c) = -1/c^2 if gamma == 1, else -gamma * c^(-gamma-1)
    
    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    def u_double_prime(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return -1.0 / (c * c)
        else:
            return -gamma * c ** (-gamma - 1)
    
    # Backward induction
    # V[t+1][z'][m'] is the continuation value
    # We need to compute V for t = horizon-2 down to 0
    
    # V[t][z][m] = max_c { u(c) + beta * sum_{z'} P(z'|z) * V[t+1][z'][R*(m-c)+income[z']] }
    
    # We'll store value functions on the grid
    V = np.zeros((horizon + 1, n_z, n_m))
    
    # Terminal: V[horizon][z][m] = 0 (no future utility)
    # Actually at t = horizon-1, we consume all, so:
    # V[horizon-1][z][m] = u(m)
    # Let me restructure: V[t] is the value at time t given resources m and state z
    
    # V[horizon-1][z][m] = u(m) (consume all)
    if gamma == 1:
        V[horizon - 1, :, :] = np.log(np.maximum(m_grid, 1e-12))[np.newaxis, :]
    else:
        V[horizon - 1, :, :] = (np.maximum(m_grid, 1e-12) ** (1 - gamma) / (1 - gamma))[np.newaxis, :]
    
    # For t = horizon-2 down to 0:
    # FOC: u'(c) = beta * sum_{z'} P(z'|z) * u'(c') * R
    # where c' is optimal consumption next period given m' = R*(m-c) + income[z']
    
    # By envelope theorem: dV[t+1][z'][m']/dm' = beta * u'(c_{t+1}^*[m', z']) ... wait
    # Actually V[t+1][z'][m'] = max_c' { u(c') + beta * E[V[t+2]] }
    # So dV[t+1][z'][m']/dm' = beta * sum_{z''} P(z''|z') * R * u'(c_{t+2}^*[m'', z''])
    # Hmm, this is getting complex. Let me use a different approach.
    
    # Actually, the FOC at time t is:
    # u'(c) = beta * sum_{z'} P(z'|z) * (dV[t+1][z']/dm')[m' = R*(m-c)+income[z']] * R
    # 
    # And by the envelope theorem for V[t+1]:
    # dV[t+1][z']/dm' = beta * sum_{z''} P(z''|z') * R * u'(c_{t+1}^*[m'', z''])
    # 
    # This is recursive. Instead, let me just store the marginal value function.
    # 
    # Let MW[t][z][m] = dV[t][z][m]/dm (marginal value of resources)
    # 
    # At t = horizon-1: V = u(m), so MW = u'(m)
    # At t = horizon-2: 
    #   V[t][z][m] = u(c) + beta * sum_{z'} P(z'|z) * V[t+1][z'][R*(m-c)+income[z']]
    #   FOC: u'(c) = beta * sum_{z'} P(z'|z) * MW[t+1][z'][R*(m-c)+income[z']] * R
    #   MW[t][z][m] = beta * sum_{z'} P(z'|z) * MW[t+1][z'][R*(m-c)+income[z']] * R
    #                 = u'(c) (by FOC)
    
    # So MW[t][z][m] = u'(c_t^*[m, z])
    
    # This means: at each t, the FOC is:
    # u'(c) = beta * R * sum_{z'} P(z'|z) * MW[t+1][z'][R*(m-c)+income[z']]
    # where MW[t+1][z'][m'] = u'(c_{t+1}^*[m', z'])
    
    # So I need to:
    # 1. Compute consumption at t+1 for all m', z'
    # 2. Compute MW[t+1][z'][m'] = u'(c_{t+1}^*[m', z'])
    # 3. Interpolate MW[t+1] onto a fine grid for the m' values
    # 4. Solve FOC for c at each m, z at time t
    # 5. Set consumption[t][z][m] = c
    # 6. MW[t][z][m] = u'(c)
    
    # MW at terminal
    MW = np.zeros((horizon, n_z, n_m))
    MW[horizon - 1, :, :] = u_prime(m_grid)[np.newaxis, :]
    
    # Fine grid for interpolation
    fine_grid = np.geomspace(0.0001, 100, 10001)
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            # Not enough time, use heuristic for remaining periods
            for tt in range(t, -1, -1):
                # Heuristic: consume fraction based on beta
                frac = 1.0 - beta
                consumption[tt, :, :] = frac * m_grid[np.newaxis, :]
                consumption[tt, :, :] = np.clip(consumption[tt, :, :], 1e-12, m_grid[np.newaxis, :])
                MW[tt, :, :] = u_prime(consumption[tt, :, :])
            break
        
        # For each z, compute MW[t+1][z] interpolated on fine grid
        # Then solve FOC for each m, z
        
        # Interpolate MW[t+1] for each z
        MW_interp = np.zeros((n_z, len(fine_grid)))
        for z in range(n_z):
            # MW[t+1][z] is defined on m_grid
            # Interpolate to fine_grid
            # Need to handle: m' can be outside m_grid range
            # Clamp to grid bounds
            m_lo = m_grid[0]
            m_hi = m_grid[-1]
            MW_interp[z, :] = np.interp(fine_grid, m_grid, MW[t + 1, z, :],
                                         left=MW[t + 1, z, 0], right=MW[t + 1, z, -1])
        
        # Now solve FOC for each z, m:
        # u'(c) = beta * R * sum_{z'} P(z'|z) * MW_interp[z'](R*(m-c)+income[z'])
        
        # Use Newton's method or bisection
        # f(c) = u'(c) - beta * R * sum_{z'} P(z'|z) * MW_interp[z'](R*(m-c)+income[z']) = 0
        # f is decreasing in c (u'(c) decreases, and as c increases, m' decreases, 
        #   MW[m'] increases since MW is decreasing in m... wait)
        
        # Actually MW is decreasing in m (more resources means lower marginal value)
        # As c increases, m' = R*(m-c)+income[z'] decreases
        # So MW[m'] increases (since m' is smaller, marginal value is higher)
        # So the RHS increases as c increases
        # LHS u'(c) decreases as c increases
        # So f(c) = LHS - RHS is decreasing in c
        # f(0) > 0 (u'(0) = inf or very large, RHS is finite)
        # f(m) < 0 (u'(m) is small, RHS with m'=income is large)
        
        # Wait, let me reconsider. MW is the marginal value of resources.
        # More resources -> lower marginal value. So MW is decreasing in m.
        # As c increases, m' decreases, so MW[m'] increases.
        # RHS = beta * R * sum P * MW[m'] increases with c.
        # LHS = u'(c) decreases with c.
        # f(c) = LHS - RHS: at c=0, LHS is large, RHS is moderate -> f > 0
        # At c=m, LHS is small, RHS is large -> f < 0
        # So f is decreasing, unique root. Good for bisection.
        
        for z in range(n_z):
            for i_m in range(n_m):
                m_val = m_grid[i_m]
                
                # Bisection for c in (0, m_val)
                c_lo = 1e-12
                c_hi = m_val
                
                # If m_val is very small, just consume it all
                if m_val < 1e-6:
                    c_opt = m_val
                    consumption[t, z, i_m] = c_opt
                    MW[t, z, i_m] = u_prime(c_opt)
                    continue
                
                # Bisection
                for _ in range(60):
                    c_mid = (c_lo + c_hi) / 2.0
                    # Compute LHS
                    lhs = u_prime(c_mid)
                    # Compute RHS
                    rhs = 0.0
                    for zp in range(n_z):
                        mp = R * (m_val - c_mid) + income[zp]
                        mp_clamped = np.clip(mp, m_grid[0], m_grid[-1])
                        mw_val = np.interp(mp_clamped, m_grid, MW[t + 1, zp, :])
                        rhs += transition[z, zp] * mw_val
                    rhs *= beta * R
                    
                    if lhs > rhs:
                        c_lo = c_mid
                    else:
                        c_hi = c_mid
                
                c_opt = (c_lo + c_hi) / 2.0
                c_opt = np.clip(c_opt, 1e-12, m_val)
                consumption[t, z, i_m] = c_opt
                MW[t, z, i_m] = u_prime(c_opt)
        
        # Check deadline
        if time.time() > deadline - 10:
            # Use heuristic for remaining
            for tt in range(t - 1, -1, -1):
                frac = 1.0 - beta
                consumption[tt, :, :] = frac * m_grid[np.newaxis, :]
                consumption[tt, :, :] = np.clip(consumption[tt, :, :], 1e-12, m_grid[np.newaxis, :])
                MW[tt, :, :] = u_prime(consumption[tt, :, :])
            break
    
    # Ensure last period is exact
    consumption[horizon - 1, :, :] = m_grid[np.newaxis, :]
    
    # Validate: 0 < c <= m + 1e-10
    consumption = np.clip(consumption, 1e-12, m_grid[np.newaxis, np.newaxis, :])
    
    return consumption
