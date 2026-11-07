"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 120  # Safety margin below 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    m_grid = resource_grid  # np.geomspace(0.001, 40, 101), shape (101,)
    n_m = len(m_grid)
    n_z = len(income)
    
    # Utility functions
    if abs(gamma - 1.0) < 1e-10:
        def util(c):
            c = np.maximum(c, 1e-12)
            return np.log(c)
        def util_prime(c):
            c = np.maximum(c, 1e-12)
            return 1.0 / c
        def util_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -1.0 / (c * c)
    else:
        def util(c):
            c = np.maximum(c, 1e-12)
            return (c ** (1 - gamma) - 1) / (1 - gamma)
        def util_prime(c):
            c = np.maximum(c, 1e-12)
            return c ** (-gamma)
        def util_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -gamma * c ** (-gamma - 1)
    
    # Value function storage: V[t, z, m]
    # We'll store V as a list of arrays, one per time period
    # V[t][z] is an array of shape (n_m,)
    
    # Terminal: at t = horizon, V = 0 (no more consumption)
    # But actually, let's think about this more carefully.
    # The problem has horizon = 6, so t = 0, 1, 2, 3, 4, 5.
    # At t = 5 (last period), c = m, V = util(m).
    # For t < 5, V[t, z, m] = max_c { util(c) + beta * E[V[t+1, z', R*(m-c) + income[z']]] }
    
    # Let's define V_next[z] = V[t+1, z, :] for all z
    # Start with V_next = terminal value
    
    # Terminal value: V_term[z, m] = util(m) for all z
    V_term = util(m_grid).copy()  # shape (n_m,)
    V_next = np.tile(V_term, (n_z, 1))  # shape (n_z, n_m)
    
    # Policy storage
    c_policy = np.zeros((horizon, n_z, n_m))
    
    # Last period: c = m
    c_policy[horizon - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    # Backward induction from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10:
            # Not enough time, fill remaining with heuristic
            for tt in range(t, -1, -1):
                for z in range(n_z):
                    c_policy[tt, z, :] = m_grid
            break
        
        V_curr = np.zeros((n_z, n_m))
        
        for z in range(n_z):
            for i, m in enumerate(m_grid):
                # Use FOC-based approach with Newton's method
                # FOC: u'(c) = beta * R * E[u'(c') * V_m'(m')]
                # where m' = R*(m-c) + income[z']
                
                # First, get a good initial guess
                # For CRRA with log utility, there's an analytical solution
                # Let's try a simple heuristic first, then refine
                
                # Heuristic: save a fraction based on beta
                # For log utility, optimal saving fraction depends on beta and R
                # c = m / (1 + beta * R * E[1/(R*(m-c) + income[z'])])
                # This is implicit, so let's use iteration
                
                # Start with c = m * (1 - beta) as initial guess (rough)
                c_guess = m * 0.5  # Start with 50% consumption
                
                # Newton's method on the FOC
                # f(c) = u'(c) - beta * R * sum_z' trans[z,z'] * u'(c') * dV/dm'(m')
                # where c' is next period's consumption, which we approximate
                
                # Actually, let's use the envelope theorem approach:
                # V_m(m, z) = u'(c*(m,z))
                # So we need to solve: u'(c) = beta * R * sum_z' trans[z,z'] * V_m(m', z')
                # where m' = R*(m-c) + income[z']
                
                # V_m(m', z') = u'(c*(m', z')) from the next period's policy
                # We can approximate this using the value function's derivative
                
                # Let's use a simpler approach: iterate on c
                # c_{k+1} = m - (1/R) * (V_next_expected / (beta * R) - income[z'])
                
                # Actually, let's just do a refined grid search with interpolation
                # but use the FOC to narrow down the search range
                
                # Get the marginal value at current m for each z'
                # V_m(m, z') ≈ (V(m+dm, z') - V(m-dm, z')) / (2*dm)
                dm = 0.01
                V_m_next = np.zeros(n_z)
                for zp in range(n_z):
                    m_plus = min(m + dm, m_grid[-1])
                    m_minus = max(m - dm, m_grid[0])
                    if m_plus == m_minus:
                        V_m_next[zp] = 0
                    else:
                        V_m_next[zp] = (V_next[zp, np.searchsorted(m_grid, m_plus)] - 
                                         V_next[zp, np.searchsorted(m_grid, m_minus)]) / (m_plus - m_minus)
                
                # Expected marginal value
                EV_m = np.dot(transition[z, :], V_m_next)
                
                # FOC: u'(c) = beta * R * EV_m
                # For log utility: 1/c = beta * R * EV_m => c = 1/(beta * R * EV_m)
                # For CRRA: c^(-gamma) = beta * R * EV_m => c = (beta * R * EV_m)^(-1/gamma)
                
                if EV_m > 0:
                    if abs(gamma - 1.0) < 1e-10:
                        c_foc = 1.0 / (beta * R * EV_m)
                    else:
                        c_foc = (beta * R * EV_m) ** (-1.0 / gamma)
                else:
                    c_foc = m * 0.5
                
                # Clip to valid range
                c_foc = np.clip(c_foc, 1e-6, m)
                
                # Refine with a local grid search around c_foc
                # Search in [max(1e-6, c_foc*0.5), min(m, c_foc*2)]
                c_lo = max(1e-6, c_foc * 0.5)
                c_hi = min(m, c_foc * 2.0)
                
                # If the range is too narrow, expand it
                if c_hi - c_lo < 0.01:
                    c_lo = max(1e-6, m * 0.1)
                    c_hi = min(m, m * 0.9)
                
                # Local grid search
                n_local = min(budget, 50)
                c_vals = np.linspace(c_lo, c_hi, n_local)
                c_vals = np.clip(c_vals, 1e-6, m)
                
                best_val = -np.inf
                best_c = c_foc
                
                for c in c_vals:
                    m_save = m - c
                    if m_save < -1e-10:
                        continue
                    
                    # Expected continuation value
                    cont_val = 0.0
                    for zp in range(n_z):
                        prob = transition[z, zp]
                        if prob < 1e-10:
                            continue
                        m_next = R * m_save + income[zp]
                        
                        # Interpolate V_next at m_next
                        if m_next <= m_grid[0]:
                            v_interp = V_next[zp, 0]
                        elif m_next >= m_grid[-1]:
                            v_interp = V_next[zp, -1]
                        else:
                            v_interp = np.interp(m_next, m_grid, V_next[zp])
                        
                        cont_val += prob * v_interp
                    
                    total_val = util(c) + beta * cont_val
                    
                    if total_val > best_val:
                        best_val = total_val
                        best_c = c
                
                c_policy[t, z, i] = best_c
                V_curr[z, i] = best_val
        
        V_next = V_curr
    
    # Enforce constraints
    for t in range(horizon):
        for z in range(n_z):
            for i in range(n_m):
                m_val = m_grid[i]
                c_val = c_policy[t, z, i]
                if c_val <= 0:
                    c_policy[t, z, i] = 1e-12
                if c_val > m_val + 1e-10:
                    c_policy[t, z, i] = m_val
    
    # Final period must equal grid within 1e-9
    c_policy[horizon - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    return c_policy
# EVOLVE-BLOCK-END
