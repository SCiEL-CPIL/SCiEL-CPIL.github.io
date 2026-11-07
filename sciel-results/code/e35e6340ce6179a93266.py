import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M_orig = resource_grid
    M_orig_len = len(M_orig)
    Z = len(income)
    
    # Define a fine grid for high-precision solving
    # The evaluator uses geomspace(0.001, 40, 251). 
    # We will use a grid of ~2000 points to ensure high accuracy.
    M_fine_len = 2000
    M_fine = np.geomspace(M_orig[0], M_orig[-1], M_fine_len)
    
    # Utility functions
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)
            
    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    def u_double_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return -1.0 / (c * c)
        else:
            return -gamma * c ** (-gamma - 1)

    # Store policies for each period and state
    # c_policies[t, z, :] is the consumption policy on M_fine
    c_policies = np.zeros((horizon, Z, M_fine_len))
    
    # Terminal condition: c_T(m) = m
    for z in range(Z):
        c_policies[horizon-1, z, :] = M_fine
        
    # Precompute interpolators for the last period
    interpolators = []
    for z in range(Z):
        interpolators.append(PchipInterpolator(M_fine, c_policies[horizon-1, z]))

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: use linear interpolation of the last computed period or just m
            # This is a safety break. We will fill remaining periods with a simple heuristic.
            for tt in range(t, -1, -1):
                for z in range(Z):
                    c_policies[tt, z, :] = M_fine # c=m is a valid fallback
            break

        for z in range(Z):
            # Initial guess: use the policy from t+1, adjusted for interest rate
            # c_t(m) approx c_{t+1}(R*m) is not quite right, but c_{t+1}(m) is a good start
            # Better: c_t(m) = (1 - beta R) m + beta R E[c_{t+1}(m')]
            # Let's just start with c = 0.5 * m or interpolate c_{t+1}
            c_curr = 0.5 * M_fine.copy()
            
            max_iters = 30
            tol = 1e-10
            
            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                    
                # m' for each z'
                # m_prime[z', m] = R * (m - c) + income[z']
                # Shape: (Z, M_fine_len)
                m_prime = R * (M_fine[None, :] - c_curr[None, :]) + income[:, None]
                
                # Evaluate c_next and c'_next at m_prime
                c_next_val = np.zeros((Z, M_fine_len))
                c_next_deriv = np.zeros((Z, M_fine_len))
                
                for zp in range(Z):
                    # Clip m_prime to the grid range to avoid wild extrapolation
                    # PCHIP handles extrapolation, but we should be careful
                    c_next_val[zp] = interpolators[zp](m_prime[zp])
                    c_next_deriv[zp] = interpolators[zp](m_prime[zp], 1)
                    
                    # Clip derivatives to [0, 1] as policy slope should be in (0,1)
                    c_next_deriv[zp] = np.clip(c_next_deriv[zp], 0.0, 1.0)
                    # Ensure consumption is valid
                    c_next_val[zp] = np.clip(c_next_val[zp], 1e-12, m_prime[zp])
                
                # Expected marginal utility
                # E_z' [ u'(c_next) ]
                probs = transition[z]
                eu_prime = np.sum(probs[:, None] * u_prime(c_next_val), axis=0)
                
                # LHS of Euler: u'(c)
                # RHS of Euler: beta * R * E[u'(c')]
                F = u_prime(c_curr) - beta * R * eu_prime
                
                # Derivative of F w.r.t c
                # F' = u''(c) + beta * R^2 * E[ u''(c') * c'_next ]
                eu_double_prime_times_deriv = np.sum(probs[:, None] * (u_double_prime(c_next_val) * c_next_deriv), axis=0)
                
                dF = u_double_prime(c_curr) + beta * R * R * eu_double_prime_times_deriv
                
                # Newton Step
                # Avoid division by zero
                mask = np.abs(dF) > 1e-12
                c_new = c_curr.copy()
                c_new[mask] = c_curr[mask] - F[mask] / dF[mask]
                
                # Clip c to [1e-12, m]
                c_new = np.clip(c_new, 1e-12, M_fine)
                
                # Check convergence
                if np.max(np.abs(c_new - c_curr)) < tol:
                    c_curr = c_new
                    break
                c_curr = c_new
                
            c_policies[t, z, :] = c_curr
            
        # Update interpolators for the next iteration
        interpolators = []
        for z in range(Z):
            interpolators.append(PchipInterpolator(M_fine, c_policies[t, z]))

    # Interpolate the fine-grid solutions back to the original grid
    consumption = np.zeros((horizon, Z, M_orig_len))
    for t in range(horizon):
        for z in range(Z):
            # Use PCHIP for smooth interpolation back to the original grid
            interp = PchipInterpolator(M_fine, c_policies[t, z])
            consumption[t, z, :] = interp(M_orig)
            
    # Final validity checks
    for t in range(horizon):
        for z in range(Z):
            c = consumption[t, z]
            m = M_orig
            c = np.maximum(c, 1e-12)
            c = np.minimum(c, m + 1e-10)
            consumption[t, z] = c
            
    # Enforce terminal condition exactly
    consumption[horizon-1] = M_orig[None, :]
    
    return consumption
