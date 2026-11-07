import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

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
        def inv_utility_prime(x):
            return 1.0 / np.maximum(x, 1e-12)
    else:
        def utility(c):
            c = np.maximum(c, 1e-12)
            return c**(1 - gamma) / (1 - gamma)
        def utility_prime(c):
            c = np.maximum(c, 1e-12)
            return c**(-gamma)
        def utility_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -gamma * c**(-gamma - 1)
        def inv_utility_prime(x):
            return np.maximum(x, 1e-12)**(-1.0/gamma)

    # Use a fine grid for m values
    m_fine = np.geomspace(0.001, 40, 1201)
    n_fine = len(m_fine)
    
    # V[t][z] will be a 1D array of length n_fine
    V = np.zeros((horizon, Z, n_fine))
    
    # Terminal: V[horizon-1][z][i] = u(m_fine[i])
    for z in range(Z):
        V[horizon-1, z, :] = utility(m_fine)
    
    # c_policy[t][z][i] stores the optimal consumption at m_fine[i] for state z at time t
    c_policy = np.zeros((horizon, Z, n_fine))
    
    # Terminal: consume all
    for z in range(Z):
        c_policy[horizon-1, z, :] = m_fine.copy()
    
    # Work backwards from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        for z in range(Z):
            c_t_z = np.zeros(n_fine)
            V_t_z = np.zeros(n_fine)
            
            for i in range(n_fine):
                m_val = m_fine[i]
                
                # Initial guess: proportional consumption
                c = max(0.5 * m_val, 1e-8)
                c = min(c, m_val * 0.999)
                
                # Newton's method on the Euler equation
                for newton_iter in range(20):
                    # Compute F(c) = u'(c) - beta * R * sum_{z'} P[z,z'] * u'(c'(m', z'))
                    F = 0.0
                    for zp in range(Z):
                        m_prime = R * (m_val - c) + income[zp]
                        m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                        c_next = np.interp(m_prime, m_fine, c_policy[t+1, zp, :])
                        c_next = max(c_next, 1e-12)
                        F += transition[z, zp] * utility_prime(c_next)
                    F = utility_prime(c) - beta * R * F
                    
                    # Compute F'(c) using numerical differentiation
                    dc = max(c * 1e-6, 1e-12)
                    c_plus = min(c + dc, m_val * (1 - 1e-10))
                    c_minus = max(c - dc, 1e-12)
                    
                    F_plus = 0.0
                    for zp in range(Z):
                        m_prime_plus = R * (m_val - c_plus) + income[zp]
                        m_prime_plus = np.clip(m_prime_plus, m_fine[0], m_fine[-1])
                        c_next_plus = np.interp(m_prime_plus, m_fine, c_policy[t+1, zp, :])
                        c_next_plus = max(c_next_plus, 1e-12)
                        F_plus += transition[z, zp] * utility_prime(c_next_plus)
                    F_plus = utility_prime(c_plus) - beta * R * F_plus
                    
                    F_minus = 0.0
                    for zp in range(Z):
                        m_prime_minus = R * (m_val - c_minus) + income[zp]
                        m_prime_minus = np.clip(m_prime_minus, m_fine[0], m_fine[-1])
                        c_next_minus = np.interp(m_prime_minus, m_fine, c_policy[t+1, zp, :])
                        c_next_minus = max(c_next_minus, 1e-12)
                        F_minus += transition[z, zp] * utility_prime(c_next_minus)
                    F_minus = utility_prime(c_minus) - beta * R * F_minus
                    
                    F_prime = (F_plus - F_minus) / (c_plus - c_minus)
                    
                    if abs(F_prime) < 1e-15:
                        break
                    
                    c_new = c - F / F_prime
                    c_new = np.clip(c_new, 1e-12, m_val * (1 - 1e-10))
                    
                    if abs(c_new - c) < 1e-12:
                        c = c_new
                        break
                    c = c_new
                
                c_t_z[i] = c
                
                # Compute V_t_z[i]
                cont_val = 0.0
                for zp in range(Z):
                    m_prime = R * (m_val - c) + income[zp]
                    m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                    cont_val += transition[z, zp] * np.interp(m_prime, m_fine, V[t+1, zp, :])
                V_t_z[i] = utility(c) + beta * cont_val
            
            c_policy[t, z, :] = c_t_z
            V[t, z, :] = V_t_z
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # Extract consumption policy on the original resource_grid
    consumption = np.zeros((horizon, Z, M))
    
    # Terminal period
    for z in range(Z):
        consumption[horizon-1, z, :] = m_grid.copy()
    
    # For t = 0 to horizon-2, interpolate from the fine grid policy
    for t in range(horizon - 1):
        for z in range(Z):
            consumption[t, z, :] = np.interp(m_grid, m_fine, c_policy[t, z, :])
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid + 1e-10)
    
    # Ensure finite values
    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=m_grid, neginf=1e-12)
    
    return consumption
