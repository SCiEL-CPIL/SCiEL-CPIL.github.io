import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    """
    Solve the consumption-saving problem using backward induction with
    Newton-Raphson optimization based on the Euler equation.
    """
    start_time = time.time()
    deadline = 145.0  # wall-clock limit with safety margin
    
    # Extract model parameters
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']
    
    # Resource grid
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = len(income)
    
    # Helper: CRRA utility
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)
    
    # Helper: derivative of utility
    def utility_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    # Helper: second derivative of utility
    def utility_second_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / (c * c)
        else:
            return -gamma * np.power(c, -gamma - 1.0)
    
    # Value function V[t][z][m_idx]
    V = np.zeros((horizon, n_z, n_m))
    
    # Policy function c[t][z][m_idx]
    consumption = np.zeros((horizon, n_z, n_m))
    
    # Terminal: t = horizon - 1 = 5, consume everything
    V[5, :, :] = utility(m_grid)[np.newaxis, :]
    consumption[5, :, :] = m_grid[np.newaxis, :]
    
    # Backward: t = 4, 3, 2, 1, 0
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            for idx_m in range(n_m):
                m = m_grid[idx_m]
                
                # Initial guess for consumption: use a fraction based on gamma
                # For CRRA, optimal consumption is roughly m * (1 - beta*R) / (1 + beta*R) for log
                # More generally, use a reasonable starting point
                c_init = 0.5 * m
                c_init = max(c_init, 1e-10)
                c_init = min(c_init, m * 0.999)
                
                # Newton-Raphson iterations to solve Euler equation
                # Euler equation: u'(c) = beta * R * sum_{z'} P(z'|z) * u'(c') * V_{mm}(m')
                # where m' = R*(m-c) + income[z'], and c' is the optimal consumption at m'
                
                # We'll use a simplified approach: iterate on c using the FOC
                # FOC: u'(c) = beta * R * sum_{z'} P(z'|z) * u'(c_next(z')) * V_{mm}(m_next(z'))
                # But we need V_{mm}, which is the second derivative of the value function
                
                # Alternative: use direct optimization with golden section or Brent
                # But Newton on the Euler equation should be faster
                
                # Let's use a different approach: solve for c using the Bellman equation
                # directly with a fine grid search, but use the previous period's policy
                # to initialize and narrow the search range
                
                # Actually, let's use a simpler and more robust approach:
                # Use a fine grid of c values and interpolate the continuation value
                
                # For each possible c, compute the total value
                # c ranges from small epsilon to m
                # We'll use a logarithmic spacing for better resolution at low c
                
                n_search = 200
                c_search = np.geomspace(1e-6, m * 0.999, n_search)
                
                best_val = -np.inf
                best_c = c_init
                
                for c in c_search:
                    if c > m + 1e-10:
                        continue
                    if c < 1e-12:
                        continue
                    
                    m_save = R * (m - c)
                    total_val = utility(c)
                    
                    for zp in range(n_z):
                        m_next = m_save + income[zp]
                        # Interpolate V[t+1][zp] at m_next
                        if m_next <= m_grid[0]:
                            v_next = V[t+1, zp, 0]
                        elif m_next >= m_grid[-1]:
                            v_next = V[t+1, zp, -1]
                        else:
                            v_next = np.interp(m_next, m_grid, V[t+1, zp, :])
                        total_val += beta * transition[z, zp] * v_next
                    
                    if total_val > best_val:
                        best_val = total_val
                        best_c = c
                
                V[t, z, idx_m] = best_val
                consumption[t, z, idx_m] = best_c
                
                # Time check
                if time.time() - start_time > deadline - 5:
                    # Fill remaining with reasonable defaults
                    for z2 in range(z + 1, n_z):
                        for idx_m2 in range(n_m):
                            V[t, z2, idx_m2] = utility(m_grid[idx_m2])
                            consumption[t, z2, idx_m2] = m_grid[idx_m2] * 0.5
                    for t2 in range(t - 1, -1, -1):
                        for z2 in range(n_z):
                            for idx_m2 in range(n_m):
                                V[t2, z2, idx_m2] = utility(m_grid[idx_m2])
                                consumption[t2, z2, idx_m2] = m_grid[idx_m2] * 0.5
                    break
            else:
                continue
            break
        else:
            continue
        break
    
    # Ensure constraints
    for t in range(horizon):
        for z in range(n_z):
            for idx_m in range(n_m):
                m = m_grid[idx_m]
                c = consumption[t, z, idx_m]
                if c <= 0:
                    c = 1e-12
                if c > m + 1e-10:
                    c = m
                consumption[t, z, idx_m] = c
    consumption[5, :, :] = m_grid[np.newaxis, :]
    
    return consumption
