import numpy as np
import time
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin below 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    m_out = resource_grid
    n_out = len(m_out)
    n_z = len(income)

    # Utility functions
    def u(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c_safe)
        else:
            return (c_safe ** (1 - gamma)) / (1 - gamma)

    # Dense internal grid for accurate value function interpolation
    # Using 800 points as a balance between accuracy and speed
    n_int = 800
    m_int = np.geomspace(0.001, 40, n_int)

    # Value function V[t, z, i] for t in 0..horizon-1, z in 0..n_z-1, i in 0..n_int-1
    # Policy function C[t, z, i]
    V = np.zeros((horizon, n_z, n_int))
    C = np.zeros((horizon, n_z, n_int))

    # Terminal condition: V_T(m, z) = u(m)
    V[horizon-1, :, :] = u(m_int)
    C[horizon-1, :, :] = m_int  # c = m at terminal

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline - 10:
            # If we are running out of time, break and use what we have
            # But we need to fill all t, so this is risky.
            # Given vectorization, this should be fast enough.
            pass

        V_next = V[t+1]  # Shape (n_z, n_int)
        
        for z in range(n_z):
            V_curr_z = np.zeros(n_int)
            C_curr_z = np.zeros(n_int)
            
            for i, m_val in enumerate(m_int):
                # Define the objective function to maximize
                def objective(c, m_val=m_val):
                    if c <= 1e-12 or c > m_val + 1e-10:
                        return -1e10
                    
                    u_curr = u(c)
                    cont_val = 0.0
                    for zp in range(n_z):
                        prob = transition[z, zp]
                        if prob == 0:
                            continue
                        m_prime = R * (m_val - c) + income[zp]
                        # Clamp m_prime to valid range for interpolation
                        if m_prime < m_int[0]:
                            m_prime = m_int[0]
                        elif m_prime > m_int[-1]:
                            m_prime = m_int[-1]
                        
                        v_interp = np.interp(m_prime, m_int, V_next[zp, :])
                        cont_val += prob * v_interp
                    
                    return u_curr + beta * cont_val
                
                # Maximize the objective over c in [1e-12, m_val]
                # Use bounded scalar minimization (minimize negative objective)
                try:
                    res = minimize_scalar(
                        lambda c: -objective(c),
                        bounds=(1e-12, m_val),
                        method='bounded',
                        options={'xatol': 1e-10, 'maxiter': 100}
                    )
                    c_opt = res.x
                except Exception:
                    # Fallback to a simple grid search if optimization fails
                    c_grid = np.linspace(1e-12, m_val, 100)
                    obj_vals = np.array([objective(c) for c in c_grid])
                    c_opt = c_grid[np.argmax(obj_vals)]
                
                # Ensure c_opt is valid
                c_opt = np.clip(c_opt, 1e-12, m_val)
                
                C_curr_z[i] = c_opt
                
                # Calculate V_curr_z[i] using the optimal c
                V_curr_z[i] = objective(c_opt)

            V[t, z, :] = V_curr_z
            C[t, z, :] = C_curr_z

    # Interpolate policy from internal grid to output grid
    pol = np.zeros((horizon, n_z, n_out))
    
    for t in range(horizon):
        for z in range(n_z):
            # Use np.interp for simplicity and speed
            # C[t, z, :] is the policy on m_int
            # We want it on m_out
            pol[t, z, :] = np.interp(m_out, m_int, C[t, z, :])
            
    # Final check: ensure validity
    # 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            c = pol[t, z, :]
            # Floor at 1e-12
            c = np.maximum(c, 1e-12)
            # Cap at m
            c = np.minimum(c, m_out)
            pol[t, z, :] = c
            
    # Terminal condition check: c[5] equal to grid within 1e-9
    # We set it exactly to m_out in the init, but interpolation might have introduced errors.
    # Let's enforce it.
    pol[horizon-1, :, :] = m_out

    return pol
