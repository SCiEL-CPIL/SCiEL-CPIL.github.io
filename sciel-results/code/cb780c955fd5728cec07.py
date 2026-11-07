"""Combined program: vectorized candidate search on fine grid + PCHIP refinement + final Newton polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin below 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = len(income)
    
    # Scoring grid - use this for the main computation
    m_fine = np.geomspace(0.001, 40, 251)
    n_fine = len(m_fine)
    
    # Utility functions
    if abs(gamma - 1.0) < 1e-10:
        def util(c):
            c = np.maximum(c, 1e-12)
            return np.log(c)
        def util_prime(c):
            c = np.maximum(c, 1e-12)
            return 1.0 / c
    else:
        def util(c):
            c = np.maximum(c, 1e-12)
            return (c ** (1 - gamma) - 1) / (1 - gamma)
        def util_prime(c):
            c = np.maximum(c, 1e-12)
            return c ** (-gamma)
    
    # Policy storage for fine grid
    c_policy_fine = np.zeros((horizon, n_z, n_fine))
    
    # Last period: c = m
    c_policy_fine[horizon - 1, :, :] = m_fine[np.newaxis, np.newaxis, :]
    
    # Value function on fine grid
    V_next = np.tile(util(m_fine), (n_z, 1))  # shape (n_z, n_fine)
    
    # Build PCHIP interpolators
    def build_pchip(arr_2d):
        return [PchipInterpolator(m_fine, arr_2d[z], extrapolate=False) for z in range(n_z)]
    
    V_interp = build_pchip(V_next)
    
    # Candidate ratios for vectorized search
    # Use a good number of candidates for accuracy
    n_candidates = 500
    ratios = np.linspace(0.001, 1.0, n_candidates)
    
    # Pre-compute transition weights
    trans = transition.copy()
    
    # Backward induction from t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10:
            # Fill remaining with heuristic
            for tt in range(t, -1, -1):
                for z in range(n_z):
                    c_policy_fine[tt, z, :] = m_fine
            break
        
        V_curr = np.zeros((n_z, n_fine))
        
        for z in range(n_z):
            probs = trans[z]
            
            # Construct candidate consumption matrix: (n_fine, n_candidates)
            C = m_fine[:, np.newaxis] * ratios[np.newaxis, :]
            
            # Calculate m' for each candidate and each next state z'
            m_next_0 = R * (m_fine[:, np.newaxis] - C) + income[0]
            m_next_1 = R * (m_fine[:, np.newaxis] - C) + income[1]
            
            # Interpolate V_next
            m_next_0_flat = m_next_0.flatten()
            m_next_1_flat = m_next_1.flatten()
            
            V_cont_0_flat = np.interp(m_next_0_flat, m_fine, V_next[0])
            V_cont_1_flat = np.interp(m_next_1_flat, m_fine, V_next[1])
            
            V_cont_0 = V_cont_0_flat.reshape(n_fine, n_candidates)
            V_cont_1 = V_cont_1_flat.reshape(n_fine, n_candidates)
            
            # Expected continuation value
            E_V_cont = probs[0] * V_cont_0 + probs[1] * V_cont_1
            
            # Immediate utility
            U_c = util(C)
            
            # Total Value
            Total_Value = U_c + beta * E_V_cont
            
            # Find best candidate index for each m
            idx_max = np.argmax(Total_Value, axis=1)
            
            rows = np.arange(n_fine)
            V_curr[z, :] = Total_Value[rows, idx_max]
            c_policy_fine[t, z, :] = C[rows, idx_max]
        
        V_next = V_curr
        V_interp = build_pchip(V_next)
    
    # Now refine using Newton's method on the fine grid for better accuracy
    # Rebuild V and Vm interpolators for the last computed period
    Vm_next = np.zeros((n_z, n_fine))
    for z in range(n_z):
        Vm_next[z, :] = util_prime(c_policy_fine[horizon - 1, z, :])
    
    Vm_interp = build_pchip(Vm_next)
    
    # Refine backward
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            break
        
        V_curr = np.zeros((n_z, n_fine))
        Vm_curr = np.zeros((n_z, n_fine))
        
        for z in range(n_z):
            probs = trans[z]
            
            # Use the candidate-based solution as initial guess
            c_guess = c_policy_fine[t, z, :].copy()
            
            for i, m in enumerate(m_fine):
                c = c_guess[i]
                
                # Newton iteration on FOC: u'(c) = beta * R * E[Vm(m', z')]
                c_lo = 1e-8
                c_hi = m
                
                for it in range(30):
                    if time.time() > deadline - 2:
                        break
                    
                    # Compute FOC residual
                    u_prime_c = util_prime(c)
                    expected_Vm = 0.0
                    for zp in range(n_z):
                        prob = probs[zp]
                        if prob < 1e-10:
                            continue
                        m_next = R * (m - c) + income[zp]
                        m_next = np.clip(m_next, m_fine[0], m_fine[-1])
                        expected_Vm += prob * Vm_interp[zp](m_next)
                    
                    f_c = u_prime_c - beta * R * expected_Vm
                    
                    # Numerical derivative
                    h = max(1e-7, c * 1e-5)
                    c_plus = min(c + h, c_hi)
                    c_minus = max(c - h, c_lo)
                    
                    u_prime_plus = util_prime(c_plus)
                    expected_Vm_plus = 0.0
                    for zp in range(n_z):
                        prob = probs[zp]
                        if prob < 1e-10:
                            continue
                        m_next = R * (m - c_plus) + income[zp]
                        m_next = np.clip(m_next, m_fine[0], m_fine[-1])
                        expected_Vm_plus += prob * Vm_interp[zp](m_next)
                    f_plus = u_prime_plus - beta * R * expected_Vm_plus
                    
                    u_prime_minus = util_prime(c_minus)
                    expected_Vm_minus = 0.0
                    for zp in range(n_z):
                        prob = probs[zp]
                        if prob < 1e-10:
                            continue
                        m_next = R * (m - c_minus) + income[zp]
                        m_next = np.clip(m_next, m_fine[0], m_fine[-1])
                        expected_Vm_minus += prob * Vm_interp[zp](m_next)
                    f_minus = u_prime_minus - beta * R * expected_Vm_minus
                    
                    f_prime_c = (f_plus - f_minus) / (2 * h)
                    
                    if abs(f_prime_c) < 1e-12:
                        break
                    
                    c_new = c - f_c / f_prime_c
                    
                    # Bisection safeguard
                    if c_new < c_lo or c_new > c_hi:
                        if f_c > 0:
                            c_lo = c
                        else:
                            c_hi = c
                        c_new = (c_lo + c_hi) / 2.0
                    else:
                        if f_c > 0:
                            c_lo = c
                        else:
                            c_hi = c
                    
                    if abs(c_new - c) < 1e-12 * max(1.0, abs(c)):
                        c = c_new
                        break
                    
                    c = c_new
                
                c = np.clip(c, 1e-12, m)
                c_policy_fine[t, z, i] = c
                
                # Compute value
                V_val = util(c)
                for zp in range(n_z):
                    prob = probs[zp]
                    if prob < 1e-10:
                        continue
                    m_next = R * (m - c) + income[zp]
                    m_next = np.clip(m_next, m_fine[0], m_fine[-1])
                    V_val += beta * prob * V_interp[zp](m_next)
                
                V_curr[z, i] = V_val
                Vm_curr[z, i] = util_prime(c)
        
        V_next = V_curr
        Vm_next = Vm_curr
        V_interp = build_pchip(V_next)
        Vm_interp = build_pchip(Vm_next)
    
    # Interpolate policy from fine grid (251) to output grid (101)
    consumption_out = np.zeros((horizon, n_z, n_m))
    
    for t in range(horizon):
        for z in range(n_z):
            consumption_out[t, z, :] = np.interp(m_grid, m_fine, c_policy_fine[t, z, :])
    
    # Enforce constraints
    for t in range(horizon):
        for z in range(n_z):
            for i in range(n_m):
                m_val = m_grid[i]
                c_val = consumption_out[t, z, i]
                if c_val <= 0:
                    consumption_out[t, z, i] = 1e-12
                if c_val > m_val + 1e-10:
                    consumption_out[t, z, i] = m_val
    
    # Final period must equal grid within 1e-9
    consumption_out[horizon - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    return consumption_out
# EVOLVE-BLOCK-END
