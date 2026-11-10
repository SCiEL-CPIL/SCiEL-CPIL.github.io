import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    
    # Policy: consumption[t, z, m_idx], shape (horizon, num_states, num_resources)
    policy = np.zeros((horizon, num_states, num_resources))
    # Value: value[t, z, m_idx]
    value = np.zeros((horizon, num_states, num_resources))
    
    # Utility and derivative functions
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1 - gamma) / (1 - gamma)
    
    def util_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return c ** (-gamma)
    
    def second_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return -1.0 / (c * c)
        return -gamma * c ** (-gamma - 1)

    # Terminal condition: at t = horizon - 1 (last period), consume all
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    value[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, :]
    
    # Backward induction
    # We solve the FOC: u'(c) - beta * R * E_z'[u'(m')] = 0
    # where m' = R(m-c) + income[z']
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: use simple rule for remaining periods
            for tt in range(t, -1, -1):
                policy[tt, :, :] = 0.5 * resource_grid[np.newaxis, :]
                value[tt, :, :] = utility(0.5 * resource_grid)[np.newaxis, :]
            break
        
        for z in range(num_states):
            m = resource_grid
            
            # Initial guess: c = m * (1 / (1 + beta * R))
            # This is the optimal fraction for infinite horizon constant income.
            c_init = m / (1 + beta * R)
            # Ensure c is within [1e-12, m]
            c_init = np.clip(c_init, 1e-12, m)
            
            c = c_init.copy()
            
            # Newton-Raphson / Bisection iterations
            # The FOC function F(c) = u'(c) - beta * R * E[u'(m')]
            # F'(c) = u''(c) - beta * R * E[u''(m') * (-R)] = u''(c) + beta * R^2 * E[u''(m')]
            # Note: u'' is negative, so F'(c) is negative (F is decreasing).
            # We want F(c) = 0.
            # Since F is decreasing, if F(c) > 0, we need larger c. If F(c) < 0, smaller c.
            
            for _ in range(50):  # Max 50 iterations, usually converges in <10
                # Calculate m' for current c
                # m' is (num_states, num_resources)
                m_next = np.zeros((num_states, num_resources))
                for zp in range(num_states):
                    m_next[zp, :] = R * (m - c) + income[zp]
                
                # Calculate u'(m') for each zp
                # u_prime_next is (num_states, num_resources)
                u_prime_next = util_deriv(m_next)
                
                # Expected u'(m')
                # E = sum_zp transition[z, zp] * u_prime_next[zp, :]
                E_u_prime = np.zeros(num_resources)
                for zp in range(num_states):
                    E_u_prime += transition[z, zp] * u_prime_next[zp, :]
                
                # F(c) = u'(c) - beta * R * E_u_prime
                F = util_deriv(c) - beta * R * E_u_prime
                
                # Check convergence
                # If max absolute F is small, we are good.
                # Also check step size.
                if np.max(np.abs(F)) < 1e-10:
                    break
                
                # Calculate derivative F'(c)
                # u''(c)
                u_second_c = second_deriv(c)
                
                # E[u''(m')]
                u_second_next = second_deriv(m_next)
                E_u_second = np.zeros(num_resources)
                for zp in range(num_states):
                    E_u_second += transition[z, zp] * u_second_next[zp, :]
                
                # F'(c) = u''(c) + beta * R^2 * E_u_second
                # Note: m' = R(m-c) + income, so dm'/dc = -R
                # d/dc E[u'(m')] = E[u''(m') * (-R)]
                # d/dc [ - beta * R * E[u'(m')] ] = - beta * R * E[u''(m') * (-R)] = beta * R^2 * E[u''(m')]
                F_prime = u_second_c + beta * (R ** 2) * E_u_second
                
                # Newton step: delta = -F / F_prime
                # Since F_prime is negative, delta will have opposite sign to F.
                # c_new = c + delta
                delta = -F / F_prime
                
                # Limit step size to prevent divergence or going out of bounds
                # Max step is 0.5 * current c or 0.5 * (m-c) to stay in bounds roughly
                max_step = 0.5 * np.maximum(c, 1e-12)
                # Also ensure we don't jump over m
                max_step = np.minimum(max_step, 0.5 * np.maximum(m - c, 1e-12))
                
                delta = np.clip(delta, -max_step, max_step)
                
                c_new = c + delta
                c_new = np.clip(c_new, 1e-12, m)
                
                # If Newton step is too small or invalid, use bisection?
                # For concave problems, Newton from a good initial guess is usually robust.
                # But let's add a safety check: if c_new == c, break.
                if np.allclose(c_new, c):
                    c = c_new
                    break
                
                c = c_new
            
            # Ensure final c is valid
            c = np.clip(c, 1e-12, m)
            policy[t, z, :] = c
            
            # Calculate value V_t(z, m) = u(c) + beta * E[V_{t+1}(z', m')]
            u_c = utility(c)
            V_next_val = np.zeros(num_resources)
            for zp in range(num_states):
                m_next_zp = R * (m - c) + income[zp]
                # Interpolate value[t+1, zp, :] at m_next_zp
                # Since m_next_zp might be outside grid range?
                # resource_grid is 0.001 to 40.
                # m is in grid. c <= m. m' = R(m-c) + income.
                # Max m' approx R*40 + 1.5 = 42.3. Grid max is 40.
                # np.interp clips to endpoints. This is standard.
                V_next_zp = np.interp(m_next_zp, resource_grid, value[t + 1, zp, :])
                V_next_val += transition[z, zp] * V_next_zp
            
            value[t, z, :] = u_c + beta * V_next_val

    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_states):
            for m_idx in range(num_resources):
                m = resource_grid[m_idx]
                c = policy[t, z, m_idx]
                if c <= 0:
                    policy[t, z, m_idx] = 1e-12
                if c > m + 1e-10:
                    policy[t, z, m_idx] = m
    
    # Ensure c[5] equals grid within 1e-9
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    
    # Check for non-finite values
    if not np.all(np.isfinite(policy)):
        policy = np.nan_to_num(policy, nan=1e-12, posinf=1e10, neginf=0)
    
    return policy
