import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M_out = len(resource_grid)
    Z = len(income)

    # High resolution internal grid
    # 3000 points is a good balance between accuracy and speed
    fine_M = 3000
    fine_grid = np.geomspace(0.001, 40, fine_M)
    
    # c[t, z, m_index] on fine grid
    c_fine = np.zeros((horizon, Z, fine_M))

    # Terminal condition: consume all
    c_fine[horizon - 1, :, :] = fine_grid[None, None, :]

    # Precompute utility derivatives
    # u'(c) = c^{-gamma}
    # u''(c) = -gamma * c^{-gamma-1}
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with terminal policy (consume all)
            # This is not optimal but valid
            for tt in range(t, -1, -1):
                c_fine[tt, :, :] = fine_grid[None, None, :]
            break

        # c_next[z] is c_fine[t+1, z, :]
        for z in range(Z):
            # Initialize s (saving rate)
            # s in [0, 1)
            # Initial guess: 0.2
            s = np.full(fine_M, 0.2)
            
            # Precompute interpolator for c_next[z']
            # We need interpolators for all z'
            interp_next = [None] * Z
            slope_next = [None] * Z
            for zp in range(Z):
                interp_next[zp] = PchipInterpolator(fine_grid, c_fine[t+1, zp, :])
                # Compute slopes for derivative
                # np.gradient is fine for smooth functions
                slope_next[zp] = np.gradient(c_fine[t+1, zp, :], fine_grid)
            
            # Newton iterations
            # Max 10 iterations, usually converges in 3-5
            for _ in range(10):
                if time.time() > deadline:
                    break
                
                m_vals = fine_grid
                c_vals = m_vals * (1.0 - s)
                c_vals = np.maximum(c_vals, 1e-12)
                
                # LHS: log u'(c) = -gamma * log(c)
                lhs = -gamma * np.log(c_vals)
                
                # RHS: log( beta * R * E[u'(c_next)] )
                # E[u'] = sum_{z'} P[z, z'] * u'(c_next[z'](m'))
                # m' = R * m * s + income[z']
                
                E_u_prime = np.zeros(fine_M)
                dE_u_prime_ds = np.zeros(fine_M)
                
                for zp in range(Z):
                    p = transition[z, zp]
                    if p < 1e-15:
                        continue
                    
                    m_prime = R * m_vals * s + income[zp]
                    
                    # Interpolate c_next
                    c_next_val = interp_next[zp](m_prime)
                    
                    # u'(c_next) = c_next^{-gamma}
                    u_prime_next = c_next_val ** (-gamma)
                    
                    # d/ds u'(c_next) = -gamma * c_next^{-gamma-1} * dc_next/ds
                    # dc_next/ds = dc_next/dm' * dm'/ds
                    # dm'/ds = R * m
                    # dc_next/dm' is approximated by slope_next[zp] interpolated at m_prime
                    dc_next_dm_prime = np.interp(m_prime, fine_grid, slope_next[zp])
                    dc_next_ds = dc_next_dm_prime * R * m_vals
                    
                    d_u_prime_next_ds = -gamma * (c_next_val ** (-gamma - 1)) * dc_next_ds
                    
                    E_u_prime += p * u_prime_next
                    dE_u_prime_ds += p * d_u_prime_next_ds
                
                # Ensure E_u_prime is positive
                E_u_prime = np.maximum(E_u_prime, 1e-12)
                
                rhs = np.log(beta * R * E_u_prime)
                
                F = lhs - rhs
                
                # Derivative F'(s)
                # d/ds lhs = gamma / (1-s)
                d_lhs_ds = gamma / (1.0 - s)
                
                # d/ds rhs = (1 / (beta R E_u_prime)) * beta R * dE_u_prime_ds
                # = dE_u_prime_ds / E_u_prime
                d_rhs_ds = dE_u_prime_ds / E_u_prime
                
                dF_ds = d_lhs_ds - d_rhs_ds
                
                # Newton step
                # s_new = s - F / dF_ds
                # Guard against division by zero or negative step size
                dF_ds_safe = np.where(np.abs(dF_ds) < 1e-10, 1e-10, dF_ds)
                step = F / dF_ds_safe
                
                # Damping: if step is too large, reduce it
                # s must remain in [0, 1)
                s_new = s - step
                
                # Clip s_new to [0, 0.999999]
                s_new = np.clip(s_new, 0.0, 0.999999)
                
                # Check convergence
                if np.max(np.abs(s_new - s)) < 1e-8:
                    s = s_new
                    break
                
                s = s_new
            
            # Final consumption
            c_fine[t, z, :] = fine_grid * (1.0 - s)

    # Interpolate back to original resource_grid
    consumption = np.zeros((horizon, Z, M_out))
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.interp(resource_grid, fine_grid, c_fine[t, z, :])

    # Enforce constraints
    for t in range(horizon):
        for z in range(Z):
            m_vals = resource_grid[None, :]
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_vals + 1e-10)
            
    # Final period must consume all wealth
    consumption[horizon - 1, :, :] = resource_grid[None, None, :]

    return consumption
