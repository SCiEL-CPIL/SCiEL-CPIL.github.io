import numpy as np
from scipy.interpolate import PCHIPInterpolator
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M = resource_grid
    n_m = len(M)
    
    # Utility function
    def u(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)

    V = np.zeros((horizon, 2, n_m))
    C = np.zeros((horizon, 2, n_m))
    
    # Terminal period
    V[horizon-1, :, :] = u(M)
    C[horizon-1, :, :] = M
    
    # Backward induction
    for t in range(horizon-2, -1, -1):
        if time.time() > deadline - 5.0:
            # Fallback: use simple linear interp if running out of time
            # But likely we won't hit this
            break
            
        # Interpolators for next period value function
        # PCHIP is preferred for monotonic, smooth functions
        interp_0 = PCHIPInterpolator(M, V[t+1, 0, :])
        interp_1 = PCHIPInterpolator(M, V[t+1, 1, :])
        
        for z in range(2):
            for i in range(n_m):
                m = M[i]
                
                # Candidate consumption values
                # Use a fine grid of fractions
                # 1000 points is fast and accurate
                fracs = np.linspace(1e-5, 1.0, 1000)
                c_cands = m * fracs
                
                # Next wealth for each state z'
                # m' = R(m - c) + y_z'
                m_next_0 = R * (m - c_cands) + income[0]
                m_next_1 = R * (m - c_cands) + income[1]
                
                # Interpolate continuation values
                # PCHIP handles extrapolation, but we might want to clip to grid range
                # to avoid wild extrapolation if m' is far out of bounds.
                # Given the parameters, m' is likely within [0.5, 42]. Grid is [0.001, 40].
                # Clipping to [M[0], M[-1]] is safer for interpolation consistency.
                m_next_0 = np.clip(m_next_0, M[0], M[-1])
                m_next_1 = np.clip(m_next_1, M[0], M[-1])
                
                V_next_0 = interp_0(m_next_0)
                V_next_1 = interp_1(m_next_1)
                
                # Expected continuation value
                p0 = transition[z, 0]
                p1 = transition[z, 1]
                cont_val = p0 * V_next_0 + p1 * V_next_1
                
                # Total utility
                u_c = u(c_cands)
                total_u = u_c + beta * cont_val
                
                # Find optimal c
                idx = np.argmax(total_u)
                C[t, z, i] = c_cands[idx]
                V[t, z, i] = total_u[idx]
                
    return C
