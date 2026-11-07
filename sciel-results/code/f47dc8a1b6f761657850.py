import numpy as np
import time
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin below 150s
    
    # Extract model parameters
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Output grid
    n_out = len(resource_grid)
    
    # Internal fine grid for value function interpolation
    # Use a finer grid than the 251 scoring grid for better accuracy
    n_fine = 500
    m_fine = np.geomspace(0.001, 40, n_fine)
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # Utility derivative
    def utility_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # Initialize Value Function for terminal period (t = horizon)
    # V_next has shape (n_fine, 2)
    V_next = np.zeros((n_fine, 2))
    
    # Policy arrays for fine grid
    policy_fine = np.zeros((horizon, 2, n_fine))
    
    # Backward Induction
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline:
            break
            
        V_curr = np.zeros((n_fine, 2))
        
        for z in range(2):
            # Pre-compute transition probs for z
            probs = transition[z]  # [p0, p1]
            
            for i in range(n_fine):
                m = m_fine[i]
                
                # Define the Euler equation residual
                # We want to find c such that:
                # u'(c) = beta * sum_z' P(z'|z) * R * V_m(R*(m-c) + income[z'], z')
                
                def euler_residual(c):
                    if c <= 1e-12 or c >= m:
                        return np.inf
                    m_next_0 = R * (m - c) + income[0]
                    m_next_1 = R * (m - c) + income[1]
                    
                    # Interpolate value function derivatives
                    # V_m is the derivative of V with respect to m
                    # We approximate V_m using central differences on V_next
                    V0 = V_next[:, 0]
                    V1 = V_next[:, 1]
                    
                    # Compute V_m using finite differences
                    # For simplicity, use forward/backward differences
                    dV0 = np.zeros(n_fine)
                    dV1 = np.zeros(n_fine)
                    
                    # Central differences where possible
                    dV0[1:-1] = (V0[2:] - V0[:-2]) / (m_fine[2:] - m_fine[:-2])
                    dV1[1:-1] = (V1[2:] - V1[:-2]) / (m_fine[2:] - m_fine[:-2])
                    
                    # Boundary points
                    dV0[0] = (V0[1] - V0[0]) / (m_fine[1] - m_fine[0])
                    dV0[-1] = (V0[-1] - V0[-2]) / (m_fine[-1] - m_fine[-2])
                    dV1[0] = (V1[1] - V1[0]) / (m_fine[1] - m_fine[0])
                    dV1[-1] = (V1[-1] - V1[-2]) / (m_fine[-1] - m_fine[-2])
                    
                    # Interpolate V_m at m_next
                    Vm_next_0 = np.interp(m_next_0, m_fine, dV0)
                    Vm_next_1 = np.interp(m_next_1, m_fine, dV1)
                    
                    rhs = beta * (probs[0] * R * Vm_next_0 + probs[1] * R * Vm_next_1)
                    return utility_prime(c) - rhs
                
                # Solve for c using brentq
                # c is in (0, m)
                try:
                    c_opt = brentq(euler_residual, 1e-12, m - 1e-12, xtol=1e-10, rtol=1e-10)
                except:
                    # Fallback: use direct optimization on a small grid
                    c_grid = np.linspace(1e-12, m, 100)
                    vals = np.zeros(len(c_grid))
                    for j, c in enumerate(c_grid):
                        m_next_0 = R * (m - c) + income[0]
                        m_next_1 = R * (m - c) + income[1]
                        V0 = V_next[:, 0]
                        V1 = V_next[:, 1]
                        V_cont_0 = np.interp(m_next_0, m_fine, V0)
                        V_cont_1 = np.interp(m_next_1, m_fine, V1)
                        E_V = probs[0] * V_cont_0 + probs[1] * V_cont_1
                        vals[j] = utility(c) + beta * E_V
                    c_opt = c_grid[np.argmax(vals)]
                
                policy_fine[t, z, i] = c_opt
                
                # Compute V_curr at this point
                m_next_0 = R * (m - c_opt) + income[0]
                m_next_1 = R * (m - c_opt) + income[1]
                V0 = V_next[:, 0]
                V1 = V_next[:, 1]
                V_cont_0 = np.interp(m_next_0, m_fine, V0)
                V_cont_1 = np.interp(m_next_1, m_fine, V1)
                E_V = probs[0] * V_cont_0 + probs[1] * V_cont_1
                V_curr[i, z] = utility(c_opt) + beta * E_V
        
        V_next = V_curr
    
    # Interpolate policy from fine grid to output grid
    consumption_out = np.zeros((horizon, 2, n_out))
    
    for t in range(horizon):
        for z in range(2):
            consumption_out[t, z, :] = np.interp(resource_grid, m_fine, policy_fine[t, z, :])
    
    return consumption_out
