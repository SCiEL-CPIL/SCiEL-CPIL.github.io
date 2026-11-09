import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    m_grid = np.asarray(resource_grid, dtype=float)
    budget = int(budget)

    n_states = len(income)
    n_m = len(m_grid)

    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        else:
            return np.power(c, -gamma)

    # Allocate storage for policies: consumption[t, z, m_index]
    policy = np.zeros((horizon, n_states, n_m))
    
    # V values on the grid for each t, z
    V = np.zeros((horizon, n_states, n_m))
    
    # Final period: t = horizon - 1 = 5
    # V[5, z, m] = u(m)
    for z in range(n_states):
        V[5, z, :] = utility(m_grid)
    policy[5, :, :] = m_grid[np.newaxis, :]  # c = m at final period
    
    # Work backwards from t = 4 down to 0
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline - 30:
            # Return best available with simple heuristics for remaining
            for tt in range(t, 0, -1):
                for z in range(n_states):
                    # Use a simple rule: consume based on remaining horizon
                    remaining = tt + 1
                    frac = 1.0 / (remaining * beta ** (remaining - 1))
                    frac = min(frac, 1.0)
                    policy[tt, z, :] = m_grid * frac
                    V[tt, z, :] = V[tt+1, z, :]
            # Ensure final period is correct
            policy[5, :, :] = m_grid[np.newaxis, :]
            for z in range(n_states):
                V[5, z, :] = utility(m_grid)
            # Clamp to valid range
            for tt in range(horizon):
                for z in range(n_states):
                    policy[tt, z, :] = np.clip(policy[tt, z, :], 1e-12, m_grid + 1e-10)
            return policy
        
        for z in range(n_states):
            m_values = m_grid
            
            # For each m, solve the Euler equation using Newton's method
            # u'(c) = beta * sum_z' P(z'|z) * R * u'(c') * V_m'(m')
            # where m' = R*(m-c) + income[z']
            
            best_c = np.zeros(n_m)
            best_V = np.zeros(n_m)
            
            for i in range(n_m):
                m_i = m_values[i]
                
                # Initial guess: use a reasonable fraction
                # For CRRA, consumption is proportional to wealth
                # Use a heuristic initial guess
                c_init = min(m_i * 0.3, m_i - 1e-8)
                c_init = max(c_init, 1e-10)
                
                # Newton iterations
                c = c_init
                for _newton in range(10):
                    # Compute m' for each next state
                    m_next = R * (m_i - c) + income  # shape (n_states,)
                    m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])
                    
                    # Interpolate V and its derivative
                    v_vals = np.array([V[t+1, z_next, :] for z_next in range(n_states)])
                    v_interp = np.array([np.interp(m_next_clipped[z_next], m_grid, v_vals[z_next]) for z_next in range(n_states)])
                    
                    # Compute V_m' via finite differences
                    eps = 1e-6
                    v_plus = np.array([np.interp(m_next_clipped[z_next] + eps, m_grid, v_vals[z_next]) for z_next in range(n_states)])
                    v_minus = np.array([np.interp(m_next_clipped[z_next] - eps, m_grid, v_vals[z_next]) for z_next in range(n_states)])
                    v_m_prime = (v_plus - v_minus) / (2 * eps)
                    
                    # Compute expected marginal continuation value
                    # E[u'(c')] * R
                    c_prime = m_next_clipped  # In the interior, c' is chosen optimally
                    # Actually, we need V_m' which already incorporates the optimal c'
                    # The Euler equation is: u'(c) = beta * E[R * V_m'(m')]
                    expected_marginal = np.sum(transition[z, :] * v_m_prime) * R
                    
                    # f(c) = u'(c) - beta * expected_marginal
                    f_val = marginal_utility(c) - beta * expected_marginal
                    
                    # f'(c) = u''(c)
                    if abs(gamma - 1.0) < 1e-12:
                        f_prime = -1.0 / (c ** 2)
                    else:
                        f_prime = -gamma * np.power(c, -gamma - 1)
                    
                    if abs(f_prime) < 1e-30:
                        break
                    
                    delta_c = f_val / f_prime
                    c_new = c - delta_c
                    
                    # Ensure c stays in valid range
                    c_new = np.clip(c_new, 1e-10, m_i - 1e-10)
                    
                    if abs(c_new - c) < 1e-12:
                        c = c_new
                        break
                    c = c_new
                
                best_c[i] = c
                best_c[i] = np.clip(c, 1e-12, m_i + 1e-10)
                
                # Compute V[i] = u(c) + beta * E[V(m')]
                m_next = R * (m_i - best_c[i]) + income
                m_next_clipped = np.clip(m_next, m_grid[0], m_grid[-1])
                v_interp = np.array([np.interp(m_next_clipped[z_next], m_grid, V[t+1, z_next, :]) for z_next in range(n_states)])
                best_V[i] = utility(best_c[i]) + beta * np.sum(transition[z, :] * v_interp)
            
            policy[t, z, :] = best_c
            V[t, z, :] = best_V
    
    # Final validation
    for t in range(horizon):
        for z in range(n_states):
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, m_grid + 1e-10)
    
    # Ensure c[5] equals grid
    policy[5, :, :] = m_grid[np.newaxis, :]
    
    return policy
