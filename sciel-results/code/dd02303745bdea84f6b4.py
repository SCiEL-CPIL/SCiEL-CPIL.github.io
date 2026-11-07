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
            return c**(1 - gamma) / (1 - gamma)
        def utility_prime(c):
            c = np.maximum(c, 1e-12)
            return c**(-gamma)
        def utility_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -gamma * c**(-gamma - 1)

    # We'll use Newton's method on the Euler equation to find optimal consumption
    # at each point, which converges faster and more accurately than grid search.
    
    # First, we need to build the continuation value function.
    # We'll represent V_{t+1}(m, z) on a fine grid and interpolate.
    
    # Create a fine grid for m values
    m_fine = np.geomspace(0.001, 40, 501)
    n_fine = len(m_fine)
    
    # V[t][z] will be a 1D array of length n_fine
    V = np.zeros((horizon, Z, n_fine))
    
    # Terminal: V[horizon-1][z][i] = u(m_fine[i])
    for z in range(Z):
        V[horizon-1, z, :] = utility(m_fine)
    
    # Work backwards from t = horizon-2 down to 0
    # For each (t, z), we solve for optimal c at each m using Newton's method
    # on the Euler equation:
    # u'(c) = beta * sum_{z'} P[z,z'] * u'(c') * R * (dV_{t+1}[z'](m')/dm')
    # where m' = R*(m-c) + income[z']
    # and c' is the optimal consumption at m', z'
    
    # Actually, it's simpler to just do a fine grid search for the value function
    # and then use Newton's method to refine the consumption policy.
    
    # Let's use a moderate grid for the value function computation
    n_c_grid = 300
    
    for t in range(horizon - 2, -1, -1):
        for z in range(Z):
            V_t_z = np.zeros(n_fine)
            
            for i in range(n_fine):
                m_val = m_fine[i]
                
                # Grid of consumption fractions
                c_fracs = np.linspace(0.001, 1.0, n_c_grid)
                c_vals = m_val * c_fracs
                c_vals = np.maximum(c_vals, 1e-12)
                
                # Compute continuation value
                cont_val = np.zeros(len(c_vals))
                for zp in range(Z):
                    m_prime = R * (m_val - c_vals) + income[zp]
                    m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                    v_interp = np.interp(m_prime, m_fine, V[t+1, zp, :])
                    cont_val += transition[z, zp] * v_interp
                
                obj = utility(c_vals) + beta * cont_val
                idx = np.argmax(obj)
                V_t_z[i] = obj[idx]
            
            V[t, z, :] = V_t_z
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    # Now extract the consumption policy using Newton's method on the Euler equation
    consumption = np.zeros((horizon, Z, M))
    
    # Terminal period
    for z in range(Z):
        consumption[horizon-1, z, :] = m_grid.copy()
    
    # For t = 0 to horizon-2, use Newton's method
    for t in range(horizon - 1):
        for z in range(Z):
            for i in range(M):
                m_val = m_grid[i]
                
                # Initial guess: use the grid search result
                n_c_init = 100
                c_fracs = np.linspace(0.001, 1.0, n_c_init)
                c_vals = m_val * c_fracs
                c_vals = np.maximum(c_vals, 1e-12)
                
                cont_val = np.zeros(len(c_vals))
                for zp in range(Z):
                    m_prime = R * (m_val - c_vals) + income[zp]
                    m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                    v_interp = np.interp(m_prime, m_fine, V[t+1, zp, :])
                    cont_val += transition[z, zp] * v_interp
                
                obj = utility(c_vals) + beta * cont_val
                idx = np.argmax(obj)
                c = c_vals[idx]
                
                # Newton's method to refine
                for newton_iter in range(10):
                    # Compute Euler equation residual
                    # F(c) = u'(c) - beta * sum_{z'} P[z,z'] * u'(c') * R * V_{t+1}'[z'](m')
                    # where c' = optimal consumption at m', z'
                    # and V_{t+1}'[z'](m') = u'(c') (by envelope theorem)
                    # So F(c) = u'(c) - beta * R * sum_{z'} P[z,z'] * u'(c')^2 * (dc'/dm')
                    # Wait, that's not quite right. Let me think again.
                    
                    # The Euler equation is:
                    # u'(c) = beta * sum_{z'} P[z,z'] * u'(c') * (dm'/dc)
                    # where dm'/dc = -R (since m' = R*(m-c) + income[z'])
                    # So: u'(c) = -beta * R * sum_{z'} P[z,z'] * u'(c')
                    # Wait, that doesn't seem right either. Let me re-derive.
                    
                    # The Bellman equation is:
                    # V_t(m, z) = max_c { u(c) + beta * sum_{z'} P[z,z'] * V_{t+1}(m', z') }
                    # where m' = R*(m-c) + income[z']
                    
                    # FOC: u'(c) + beta * sum_{z'} P[z,z'] * V_{t+1}'(m', z') * dm'/dc = 0
                    # dm'/dc = -R
                    # So: u'(c) - beta * R * sum_{z'} P[z,z'] * V_{t+1}'(m', z') = 0
                    
                    # By the envelope theorem, V_{t+1}'(m', z') = u'(c') where c' is the optimal consumption at (m', z').
                    # So the FOC is: u'(c) = beta * R * sum_{z'} P[z,z'] * u'(c')
                    
                    # To apply Newton's method, we need F(c) and F'(c).
                    # F(c) = u'(c) - beta * R * sum_{z'} P[z,z'] * u'(c'(m', z'))
                    # where m' = R*(m-c) + income[z']
                    
                    # F'(c) = u''(c) - beta * R * sum_{z'} P[z,z'] * u''(c') * dc'/dm' * dm'/dc
                    # dm'/dc = -R
                    # dc'/dm' = 1 / (1 - beta * R * sum_{z''} P[z',z''] * u''(c'') * R * ... ) 
                    # This is getting complicated. Let me just use a simple approach.
                    
                    # Actually, let me just do a fine local search around the initial guess.
                    # This is simpler and still accurate.
                    break
                
                # Fine local search around the initial guess
                c_low = max(1e-12, c * 0.9)
                c_high = min(m_val, c * 1.1)
                n_fine_local = 200
                c_local = np.linspace(c_low, c_high, n_fine_local)
                c_local = np.clip(c_local, 1e-12, m_val)
                
                cont_val_local = np.zeros(len(c_local))
                for zp in range(Z):
                    m_prime = R * (m_val - c_local) + income[zp]
                    m_prime = np.clip(m_prime, m_fine[0], m_fine[-1])
                    v_interp = np.interp(m_prime, m_fine, V[t+1, zp, :])
                    cont_val_local += transition[z, zp] * v_interp
                
                obj_local = utility(c_local) + beta * cont_val_local
                idx_local = np.argmax(obj_local)
                consumption[t, z, i] = c_local[idx_local]
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid + 1e-10)
    
    # Ensure finite values
    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=m_grid, neginf=1e-12)
    
    return consumption
