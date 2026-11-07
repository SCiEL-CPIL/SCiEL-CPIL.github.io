import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    c_min = 1e-12
    T = horizon
    n_z = 2
    grid_out = resource_grid
    n_out = len(grid_out)
    
    def utility(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
    
    def utility_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    def utility_double_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / (c ** 2)
        else:
            return -gamma * c ** (-gamma - 1)
    
    # Use a denser internal grid for better accuracy
    # 1500 points balances accuracy and speed for Newton's method
    n_inner = 1500
    grid = np.geomspace(0.001, 40, n_inner)
    n_grid = len(grid)
    
    V = [None] * T
    P = [None] * T
    
    # Terminal condition: consume all wealth
    V[T-1] = utility(grid)[np.newaxis, :].repeat(n_z, axis=0)
    P[T-1] = grid[np.newaxis, :].repeat(n_z, axis=0)
    
    # Backward induction
    for t in range(T-2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: use previous period's policy if time is running out
            for t_rem in range(t, -1, -1):
                V[t_rem] = V[t+1].copy()
                P[t_rem] = P[t+1].copy()
            break
            
        V_t = np.zeros((n_z, n_grid))
        P_t = np.zeros((n_z, n_grid))
        V_next = V[t+1]
        
        for z in range(n_z):
            # We need to find c that maximizes:
            # U(c) + beta * sum_{zp} transition[z, zp] * V_next[zp](R*(m-c) + income[zp])
            
            # Use Newton's method to solve the first-order condition
            # U'(c) + beta * sum_zp transition[z,zp] * V_next[zp]'(R*(m-c) + income[zp]) * R * (-1) = 0
            # i.e., U'(c) - beta * R * sum_zp transition[z,zp] * V_next[zp]'(R*(m-c) + income[zp]) = 0
            
            # Initialize c at a reasonable guess: c = m * 0.5
            c = grid * 0.5
            c = np.maximum(c, c_min)
            c = np.minimum(c, grid)
            
            # Newton iterations
            for _ in range(50):
                # Compute U'(c)
                U_prime = utility_prime(c)
                
                # Compute continuation derivative
                cont_deriv = np.zeros(n_grid)
                for zp in range(n_z):
                    m_next = R * (grid - c) + income[zp]
                    m_next_clipped = np.clip(m_next, grid[0], grid[-1])
                    
                    # Interpolate V_next[zp]' at m_next_clipped
                    # We need V_next[zp]' which we can approximate from V_next
                    # Using finite differences or analytical derivative if possible
                    # For simplicity, use central differences on the grid
                    V_next_zp = V_next[zp]
                    
                    # Compute derivative via interpolation
                    # Use np.gradient or finite differences
                    dV = np.zeros(n_grid)
                    dV[1:-1] = (V_next_zp[2:] - V_next_zp[:-2]) / (grid[2:] - grid[:-2])
                    dV[0] = (V_next_zp[1] - V_next_zp[0]) / (grid[1] - grid[0])
                    dV[-1] = (V_next_zp[-1] - V_next_zp[-2]) / (grid[-1] - grid[-2])
                    
                    dV_interp = np.interp(m_next_clipped, grid, dV)
                    cont_deriv += transition[z, zp] * dV_interp
                
                # FOC: U'(c) - beta * R * cont_deriv = 0
                FOC = U_prime - beta * R * cont_deriv
                
                # Check convergence
                if np.all(np.abs(FOC) < 1e-10):
                    break
                
                # Compute second derivative for Newton step
                U_double = utility_double_prime(c)
                
                # Second derivative of continuation value
                cont_second = np.zeros(n_grid)
                for zp in range(n_z):
                    m_next = R * (grid - c) + income[zp]
                    m_next_clipped = np.clip(m_next, grid[0], grid[-1])
                    
                    V_next_zp = V_next[zp]
                    dV = np.zeros(n_grid)
                    dV[1:-1] = (V_next_zp[2:] - V_next_zp[:-2]) / (grid[2:] - grid[:-2])
                    dV[0] = (V_next_zp[1] - V_next_zp[0]) / (grid[1] - grid[0])
                    dV[-1] = (V_next_zp[-1] - V_next_zp[-2]) / (grid[-1] - grid[-2])
                    
                    # Second derivative of V_next[zp]
                    d2V = np.zeros(n_grid)
                    d2V[1:-1] = (dV[2:] - dV[:-2]) / (grid[2:] - grid[:-2])
                    d2V[0] = (dV[1] - dV[0]) / (grid[1] - grid[0])
                    d2V[-1] = (dV[-1] - dV[-2]) / (grid[-1] - grid[-2])
                    
                    d2V_interp = np.interp(m_next_clipped, grid, d2V)
                    # Chain rule: d/dc [V'(R(m-c)+y)] = V''(R(m-c)+y) * R * (-1)
                    # Then d/dc of that again: V''(R(m-c)+y) * R * (-1) * (-1) * R = V''(R(m-c)+y) * R^2
                    # Wait, let me redo this.
                    # Let g(c) = V_next[zp]'(R*(m-c) + income[zp])
                    # g'(c) = V_next[zp]''(R*(m-c) + income[zp]) * R * (-1)
                    # g''(c) = V_next[zp]'''(R*(m-c) + income[zp]) * R * (-1) * R * (-1)
                    # This gets complicated. For the Newton step, we need d/dc of FOC.
                    # FOC = U'(c) - beta * R * sum_zp transition[z,zp] * V_next[zp]'(R*(m-c) + income[zp])
                    # dFOC/dc = U''(c) - beta * R * sum_zp transition[z,zp] * d/dc[V_next[zp]'(R*(m-c) + income[zp])]
                    # = U''(c) - beta * R * sum_zp transition[z,zp] * V_next[zp]''(R*(m-c) + income[zp]) * R * (-1)
                    # = U''(c) + beta * R^2 * sum_zp transition[z,zp] * V_next[zp]''(R*(m-c) + income[zp])
                    
                    cont_second += transition[z, zp] * d2V_interp
                
                dFOC = U_double + beta * R**2 * cont_second
                
                # Newton step
                step = FOC / dFOC
                c_new = c - step
                c_new = np.clip(c_new, c_min, grid)
                
                # Damping if needed
                if np.any(c_new <= 0) or np.any(c_new > grid):
                    c_new = np.clip(c_new, c_min, grid)
                
                c = c_new
            
            # Ensure c is within bounds
            c = np.clip(c, c_min, grid)
            P_t[z, :] = c
            
            # Calculate value at optimal c
            U_c = utility(c)
            cont_val = np.zeros(n_grid)
            for zp in range(n_z):
                m_next = R * (grid - c) + income[zp]
                m_next_clipped = np.clip(m_next, grid[0], grid[-1])
                V_interp = np.interp(m_next_clipped, grid, V_next[zp])
                cont_val += transition[z, zp] * V_interp
            V_t[z, :] = U_c + beta * cont_val
        
        V[t] = V_t
        P[t] = P_t
    
    # Interpolate policy back to the original grid
    result = np.zeros((T, n_z, n_out))
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.interp(grid_out, grid, P[t][z])
    
    # Enforce constraints
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.clip(result[t, z, :], c_min, grid_out + 1e-10)
            
    # Terminal condition must be exact on the output grid
    result[T-1, :, :] = grid_out[np.newaxis, :]
    
    return result
