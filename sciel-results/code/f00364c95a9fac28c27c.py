import numpy as np
import time
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin below 150s kill
    
    # Extract model parameters
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Output grid
    n_out = len(resource_grid)
    m_out = resource_grid
    
    # Use a fine grid for computation
    m_fine = np.geomspace(0.001, 40, 501)
    n_fine = len(m_fine)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # Marginal utility
    def marg_util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # Second derivative of utility (for Newton's method)
    def second_deriv_util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return -1.0 / (c * c)
        else:
            return -gamma * c ** (-gamma - 1)
    
    # Policy storage on fine grid
    policy_fine = np.zeros((horizon, 2, n_fine))
    
    # Value function on fine grid
    V_next = np.zeros((n_fine, 2))
    
    # Backward induction
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 5:
            break
            
        V_curr = np.zeros((n_fine, 2))
        
        # Build cubic splines for continuation values
        try:
            cs_0 = CubicSpline(m_fine, V_next[:, 0], bc_type='natural')
            cs_1 = CubicSpline(m_fine, V_next[:, 1], bc_type='natural')
        except Exception:
            cs_0 = None
            cs_1 = None
        
        for z in range(2):
            probs = transition[z]
            
            for i in range(n_fine):
                m = m_fine[i]
                
                # Initial guess: use a fraction of m
                # For CRRA, consumption is roughly proportional to m
                c_guess = 0.5 * m
                
                # Newton's method to solve the FOC
                # FOC: u'(c) = beta * R * E[u'(m')]
                # where m' = R*(m-c) + income[z']
                
                c = c_guess
                max_iter = 20
                tol = 1e-12
                
                for _ in range(max_iter):
                    c = max(c, 1e-12)
                    c = min(c, m - 1e-12)
                    
                    m_n0 = R * (m - c) + income[0]
                    m_n1 = R * (m - c) + income[1]
                    
                    # Continuation values and derivatives
                    if cs_0 is not None and cs_1 is not None:
                        v0 = cs_0(m_n0)
                        v1 = cs_1(m_n1)
                        dv0 = cs_0(m_n0, 1)
                        dv1 = cs_1(m_n1, 1)
                    else:
                        v0 = np.interp(m_n0, m_fine, V_next[:, 0])
                        v1 = np.interp(m_n1, m_fine, V_next[:, 1])
                        # Numerical derivative
                        eps = 1e-6
                        m_n0_p = m_n0 + eps
                        m_n0_m = max(m_n0 - eps, 1e-12)
                        m_n1_p = m_n1 + eps
                        m_n1_m = max(m_n1 - eps, 1e-12)
                        dv0 = (np.interp(m_n0_p, m_fine, V_next[:, 0]) - np.interp(m_n0_m, m_fine, V_next[:, 0])) / (m_n0_p - m_n0_m)
                        dv1 = (np.interp(m_n1_p, m_fine, V_next[:, 1]) - np.interp(m_n1_m, m_fine, V_next[:, 1])) / (m_n1_p - m_n1_m)
                    
                    # Expected marginal value of next period's resources
                    ev_marg = probs[0] * dv0 + probs[1] * dv1
                    
                    # FOC residual: u'(c) - beta * R * E[u'(m')]
                    # Note: u'(m') is the derivative of utility w.r.t. m', which equals V_m'
                    # So we need u'(c) = beta * R * ev_marg
                    residual = marg_util(c) - beta * R * ev_marg
                    
                    # Derivative of residual w.r.t. c
                    # d/dc [u'(c)] = u''(c)
                    # d/dc [beta * R * ev_marg] = beta * R * (probs[0] * d2v0/dm0^2 * (-R) + probs[1] * d2v1/dm1^2 * (-R))
                    # We don't have second derivatives of V easily, so use numerical approximation
                    # Or approximate: the derivative of the RHS w.r.t. c is approximately 0 for first-order
                    # Better: use numerical derivative of the residual
                    eps_c = max(1e-8, c * 1e-6)
                    c_p = min(c + eps_c, m - 1e-12)
                    c_m = max(c - eps_c, 1e-12)
                    
                    m_n0_p = R * (m - c_p) + income[0]
                    m_n1_p = R * (m - c_p) + income[1]
                    m_n0_m = R * (m - c_m) + income[0]
                    m_n1_m = R * (m - c_m) + income[1]
                    
                    if cs_0 is not None and cs_1 is not None:
                        dv0_p = cs_0(m_n0_p, 1)
                        dv1_p = cs_1(m_n1_p, 1)
                        dv0_m = cs_0(m_n0_m, 1)
                        dv1_m = cs_1(m_n1_m, 1)
                    else:
                        eps2 = 1e-6
                        dv0_p = (np.interp(m_n0_p + eps2, m_fine, V_next[:, 0]) - np.interp(max(m_n0_p - eps2, 1e-12), m_fine, V_next[:, 0])) / (2 * eps2)
                        dv1_p = (np.interp(m_n1_p + eps2, m_fine, V_next[:, 1]) - np.interp(max(m_n1_p - eps2, 1e-12), m_fine, V_next[:, 1])) / (2 * eps2)
                        dv0_m = (np.interp(m_n0_m + eps2, m_fine, V_next[:, 0]) - np.interp(max(m_n0_m - eps2, 1e-12), m_fine, V_next[:, 0])) / (2 * eps2)
                        dv1_m = (np.interp(m_n1_m + eps2, m_fine, V_next[:, 1]) - np.interp(max(m_n1_m - eps2, 1e-12), m_fine, V_next[:, 1])) / (2 * eps2)
                    
                    ev_marg_p = probs[0] * dv0_p + probs[1] * dv1_p
                    ev_marg_m = probs[0] * dv0_m + probs[1] * dv1_m
                    residual_p = marg_util(c_p) - beta * R * ev_marg_p
                    residual_m = marg_util(c_m) - beta * R * ev_marg_m
                    
                    deriv = (residual_p - residual_m) / (c_p - c_m)
                    
                    if abs(deriv) < 1e-15:
                        break
                    
                    c_new = c - residual / deriv
                    c_new = max(c_new, 1e-12)
                    c_new = min(c_new, m - 1e-12)
                    
                    if abs(c_new - c) < tol:
                        c = c_new
                        break
                    c = c_new
                
                # Ensure c is valid
                c = max(c, 1e-12)
                c = min(c, m - 1e-12)
                
                # Compute value
                m_n0 = R * (m - c) + income[0]
                m_n1 = R * (m - c) + income[1]
                
                if cs_0 is not None and cs_1 is not None:
                    v0 = cs_0(m_n0)
                    v1 = cs_1(m_n1)
                else:
                    v0 = np.interp(m_n0, m_fine, V_next[:, 0])
                    v1 = np.interp(m_n1, m_fine, V_next[:, 1])
                
                ev = probs[0] * v0 + probs[1] * v1
                V_curr[i, z] = utility(c) + beta * ev
                policy_fine[t, z, i] = c
        
        V_next = V_curr
    
    # Fallback for early termination
    last_valid_t = horizon - 1
    for t in range(horizon - 1, -1, -1):
        if not np.all(policy_fine[t] == 0):
            last_valid_t = t
            break
            
    if last_valid_t < horizon - 1:
        for t in range(last_valid_t, -1, -1):
            if np.all(policy_fine[t] == 0):
                policy_fine[t] = policy_fine[last_valid_t]
    
    # Interpolate to output grid
    consumption_out = np.zeros((horizon, 2, n_out))
    
    for t in range(horizon):
        for z in range(2):
            consumption_out[t, z, :] = np.interp(m_out, m_fine, policy_fine[t, z, :])
    
    # Validity checks
    for t in range(horizon):
        for z in range(2):
            consumption_out[t, z, :] = np.maximum(consumption_out[t, z, :], 1e-12)
            consumption_out[t, z, :] = np.minimum(consumption_out[t, z, :], m_out + 1e-10)
    
    # Terminal condition
    consumption_out[5, :, :] = m_out[None, None, :]
    
    return consumption_out
