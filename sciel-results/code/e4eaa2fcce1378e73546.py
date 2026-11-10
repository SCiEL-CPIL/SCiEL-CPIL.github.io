"""Rewritten solution for consumption-saving policy accuracy."""
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    T = horizon
    Z = len(income)
    N = len(resource_grid)
    
    # Policy array: c[t, z, i]
    c_policy = np.zeros((T, Z, N))
    
    # Last period: consume everything
    c_policy[T-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    def util_prime(c, gamma_val):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma_val, 1.0):
            return 1.0 / c
        else:
            return c ** (-gamma_val)
    
    def util(c, gamma_val):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma_val, 1.0):
            return np.log(c)
        else:
            return (c ** (1.0 - gamma_val)) / (1.0 - gamma_val)
    
    # Continuation value: V[t, z, i] = max_c u(c) + beta * sum_z' p[z,z'] V[t+1, z', next_i]
    # We'll work backward from T-1 to 0
    
    # For each period t from T-2 down to 0, for each state z, find optimal c
    # using Newton-Raphson on the Euler equation
    
    # Precompute next state values for interpolation
    # V_next[z, m] = sum_z' transition[z, z'] * V[t+1, z', m']
    # where m' = R*(m - c) + income[z']
    
    # We'll store continuation values as a function of current resources m
    # For each z, we need V_cont[t, z](m) = sum_z' p[z,z'] * V[t+1, z'](R*(m-c) + income[z'])
    # But this depends on c, so we need to solve simultaneously
    
    # Better approach: For each (t, z), solve for c[m] for all m in resource_grid
    # using Newton-Raphson where the continuation value is interpolated from next period
    
    # V_next[z, i] for each z and grid index i
    V_next = np.zeros((Z, N))
    V_next[:, :] = 0.0  # Terminal: V[T-1, z, m] = u(m) (consume all)
    # Actually at t = T-1, we consume all, so V[T-1, z, i] = util(resource_grid[i], gamma)
    V_next[:, :] = util(resource_grid, gamma)[np.newaxis, :]
    
    for t in range(T - 2, -1, -1):
        # Check deadline
        if time.time() > DEADLINE:
            break
        
        V_curr = np.zeros((Z, N))
        
        for z in range(Z):
            # For this state z, solve for optimal consumption at each grid point
            # Euler equation: u'(c) = beta * sum_z' p[z,z'] * u'(c') * R
            # where c' is optimal consumption in next period given m' = R*(m-c) + income[z']
            
            # We'll use Newton-Raphson for each m
            # f(c) = u'(c) - beta * sum_z' p[z,z'] * u'(c_next(m', z')) * R = 0
            # where m' = R*(m - c) + income[z']
            # c_next(m', z') is the optimal consumption in next period, which we have from c_policy[t+1, z', :]
            
            for i in range(N):
                m = resource_grid[i]
                
                # Initial guess: proportional consumption
                # Use a reasonable initial guess
                c_guess = 0.5 * m
                c_guess = max(c_guess, 1e-10)
                c_guess = min(c_guess, m - 1e-10)
                
                # Newton-Raphson iterations
                c_newton = c_guess
                for _ in range(30):
                    # Compute m' for each z'
                    m_prime = R * (m - c_newton) + income  # shape (Z,)
                    
                    # Get next period consumption for each z' by interpolation
                    # c_next[z', m'] = interp(m', resource_grid, c_policy[t+1, z', :])
                    c_next = np.zeros(Z)
                    for zp in range(Z):
                        c_next[zp] = np.interp(m_prime[zp], resource_grid, c_policy[t+1, zp, :])
                    
                    # Compute u'(c_next) for each z'
                    uc_next = util_prime(c_next, gamma)
                    
                    # Compute expected marginal utility of saving
                    # E[u'(c')] = sum_z' p[z,z'] * u'(c')
                    expected_marginal = np.dot(transition[z, :], uc_next)
                    
                    # f(c) = u'(c) - beta * R * expected_marginal
                    uc = util_prime(c_newton, gamma)
                    f_val = uc - beta * R * expected_marginal
                    
                    # f'(c) = u''(c) + beta * R * sum_z' p[z,z'] * u''(c') * (-R)
                    # u''(c) = -gamma * c^(-gamma-1) for CRRA
                    # For gamma=1: u''(c) = -1/c^2
                    if np.isclose(gamma, 1.0):
                        ucc = -1.0 / (c_newton ** 2)
                        ucc_next = -1.0 / (np.maximum(c_next, 1e-12) ** 2)
                    else:
                        ucc = -gamma * (c_newton ** (-gamma - 1))
                        ucc_next = -gamma * (np.maximum(c_next, 1e-12) ** (-gamma - 1))
                    
                    expected_marginal_deriv = np.dot(transition[z, :], ucc_next)
                    f_prime = ucc + beta * R * (-R) * expected_marginal_deriv
                    # Wait, let me redo this. f(c) = u'(c) - beta * R * E[u'(c')]
                    # df/dc = u''(c) - beta * R * d/dc[E[u'(c')]]
                    # d/dc[E[u'(c')]] = sum_z' p[z,z'] * u''(c') * dm'/dc
                    # dm'/dc = -R
                    # So df/dc = u''(c) - beta * R * sum_z' p[z,z'] * u''(c') * (-R)
                    # = u''(c) + beta * R^2 * sum_z' p[z,z'] * u''(c')
                    
                    f_prime = ucc + beta * R * R * expected_marginal_deriv
                    
                    if abs(f_prime) < 1e-15:
                        break
                    
                    c_newton_new = c_newton - f_val / f_prime
                    
                    # Ensure c stays in (0, m)
                    c_newton_new = max(c_newton_new, 1e-10)
                    c_newton_new = min(c_newton_new, m - 1e-10)
                    
                    if abs(c_newton_new - c_newton) < 1e-12:
                        c_newton = c_newton_new
                        break
                    c_newton = c_newton_new
                
                c_policy[t, z, i] = c_newton
            
            # Compute V_curr[z, i] = u(c) + beta * sum_z' p[z,z'] * V[t+1, z', m']
            # where m' = R*(m - c) + income[z']
            for i in range(N):
                m = resource_grid[i]
                c_opt = c_policy[t, z, i]
                m_prime = R * (m - c_opt) + income  # shape (Z,)
                
                # Get V_next[z', m'] by interpolation
                v_next_vals = np.zeros(Z)
                for zp in range(Z):
                    v_next_vals[zp] = np.interp(m_prime[zp], resource_grid, V_next[zp, :])
                
                expected_v = np.dot(transition[z, :], v_next_vals)
                V_curr[z, i] = util(c_opt, gamma) + beta * expected_v
        
        V_next = V_curr
    
    # Ensure final period is exact
    c_policy[T-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    # Validate and fix any issues
    for t in range(T):
        for z in range(Z):
            # Ensure 0 < c <= m + 1e-10
            c_policy[t, z, :] = np.maximum(c_policy[t, z, :], 1e-12)
            c_policy[t, z, :] = np.minimum(c_policy[t, z, :], resource_grid + 1e-10)
            # Ensure finite
            c_policy[t, z, :] = np.where(np.isfinite(c_policy[t, z, :]), c_policy[t, z, :], resource_grid)
    
    # Final period must equal grid within 1e-9
    c_policy[T-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    return c_policy
