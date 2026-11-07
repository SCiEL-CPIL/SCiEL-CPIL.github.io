import numpy as np

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Internal fine grid
    N_fine = 2000
    m_fine = np.geomspace(0.001, 40, N_fine)
    
    # Precompute utility and derivatives if needed, but we just evaluate U
    def U(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
            
    V = np.zeros((horizon, 2, N_fine))
    c_fine = np.zeros((horizon, 2, N_fine))
    
    # Terminal condition
    V[horizon-1, :, :] = U(m_fine)
    c_fine[horizon-1, :, :] = m_fine
    
    # Candidates for c
    # c is in [0, m]. We use a relative grid.
    # To avoid c=0 (log singular), start at 0.001 * m.
    c_rel = np.linspace(0.001, 1.0, 1000)
    
    for t in range(horizon - 2, -1, -1):
        for z in range(2):
            # m_fine shape (N_fine,)
            # c_grid shape (N_fine, 1000)
            c_grid = m_fine[:, None] * c_rel[None, :]
            
            # Compute m' for z'=0 and z'=1
            m_prime_0 = R * (m_fine[:, None] - c_grid) + income[0]
            m_prime_1 = R * (m_fine[:, None] - c_grid) + income[1]
            
            # Clip m' to be within m_fine range for interp?
            # m' can be very small or large.
            # np.interp handles out of bounds by clamping.
            
            # Interpolate V[t+1]
            V_next_0 = np.interp(m_prime_0, m_fine, V[t+1, 0, :])
            V_next_1 = np.interp(m_prime_1, m_fine, V[t+1, 1, :])
            
            # Continuation value
            cont_val = beta * (transition[z, 0] * V_next_0 + transition[z, 1] * V_next_1)
            
            # Total value
            total_val = U(c_grid) + cont_val
            
            # Find argmax
            idx_max = np.argmax(total_val, axis=1)
            
            c_opt = c_grid[np.arange(N_fine), idx_max]
            V[t, z, :] = total_val[np.arange(N_fine), idx_max]
            c_fine[t, z, :] = c_opt
            
    # Interpolate c_fine to resource_grid
    c_out = np.zeros((horizon, 2, len(resource_grid)))
    for t in range(horizon):
        for z in range(2):
            c_out[t, z, :] = np.interp(resource_grid, m_fine, c_fine[t, z, :])
            
    # Enforce terminal condition exactly
    c_out[horizon-1, :, :] = resource_grid
    
    # Ensure validity: 0 < c <= m
    for t in range(horizon):
        for z in range(2):
            c_out[t, z, :] = np.clip(c_out[t, z, :], 1e-12, resource_grid + 1e-10)
            
    return c_out
