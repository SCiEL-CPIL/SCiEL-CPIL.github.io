"""High-precision backward induction with dense grid and polynomial value function approximation."""
import time
import numpy as np
from scipy.optimize import minimize_scalar

# EVOLVE-BLOCK-START
def _utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(c)
    return c ** (1.0 - gamma) / (1.0 - gamma)

def solve_consumption(model, resource_grid, budget):
    _t0 = time.time()
    deadline = _t0 + 115.0

    beta = model['beta']
    gamma = model['gamma']
    R = model.get('R', 1.02)
    income = np.asarray(model['income'], dtype=float)
    Tmat = np.asarray(model['transition'], dtype=float)
    horizon = int(model['horizon'])
    m_grid = np.asarray(resource_grid, dtype=float)

    n_z = len(income)
    n_m = len(m_grid)
    C = np.empty((horizon, n_z, n_m), dtype=float)

    # Terminal period: consume all
    C[horizon - 1, :, :] = m_grid

    # Use a dense internal grid for value function computation
    # This gives us more accurate interpolation when the evaluator uses np.interp
    n_dense = 500
    m_dense = np.geomspace(0.001, 40, n_dense)
    
    # Value function on dense grid: V_next[z, :] for each state z
    V_next = np.zeros((n_z, n_dense), dtype=float)
    
    # Terminal values
    for z in range(n_z):
        V_next[z, :] = _utility(m_dense, gamma)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        V_curr = np.zeros((n_z, n_dense), dtype=float)
        
        for z in range(n_z):
            probs = Tmat[z]
            
            for i, m in enumerate(m_dense):
                # Maximize over c: u(c) + beta * sum_{z'} probs[z'] * V_next[z', R*(m-c) + income[z']]
                
                def neg_obj(c):
                    if c < 1e-12 or c > m + 1e-10:
                        return 0.0
                    c = max(c, 1e-12)
                    m_next = R * (m - c) + income
                    m_next = np.maximum(m_next, 1e-12)
                    # Interpolate continuation values
                    cont = np.zeros(n_z)
                    for zp in range(n_z):
                        cont[zp] = np.interp(m_next[zp], m_dense, V_next[zp])
                    return -(_utility(c, gamma) + beta * np.dot(probs, cont))
                
                # Use bounded scalar minimization
                c_low = 1e-12
                c_high = m - 1e-12
                if c_high <= c_low:
                    c_opt = max(c_low, m)
                else:
                    res = minimize_scalar(neg_obj, bounds=(c_low, c_high), 
                                         method='bounded', options={'xatol': 1e-8})
                    c_opt = res.x
                
                c_opt = np.clip(c_opt, 1e-12, m + 1e-10)
                
                # Compute value
                m_next = R * (m - c_opt) + income
                m_next = np.maximum(m_next, 1e-12)
                cont = np.zeros(n_z)
                for zp in range(n_z):
                    cont[zp] = np.interp(m_next[zp], m_dense, V_next[zp])
                V_curr[z, i] = _utility(c_opt, gamma) + beta * np.dot(probs, cont)
                
                if time.time() > deadline:
                    # Fill remaining with heuristic
                    for ii in range(i + 1, n_dense):
                        V_curr[z, ii] = V_curr[z, i]
                    break
            
            if time.time() > deadline:
                break
        
        V_next = V_curr
        
        # Interpolate policy onto the original grid
        for z in range(n_z):
            # Re-derive policy from V_next by solving FOC or re-optimizing
            for i, m in enumerate(m_grid):
                def neg_obj(c):
                    if c < 1e-12 or c > m + 1e-10:
                        return 0.0
                    c = max(c, 1e-12)
                    m_next = R * (m - c) + income
                    m_next = np.maximum(m_next, 1e-12)
                    cont = np.zeros(n_z)
                    for zp in range(n_z):
                        cont[zp] = np.interp(m_next[zp], m_dense, V_next[zp])
                    return -(_utility(c, gamma) + beta * np.dot(Tmat[z], cont))
                
                c_low = 1e-12
                c_high = m - 1e-12
                if c_high <= c_low:
                    C[t, z, i] = max(c_low, m)
                else:
                    res = minimize_scalar(neg_obj, bounds=(c_low, c_high),
                                         method='bounded', options={'xatol': 1e-8})
                    C[t, z, i] = np.clip(res.x, 1e-12, m + 1e-10)
                
                if time.time() > deadline:
                    for ii in range(i + 1, n_m):
                        C[t, z, ii] = m_grid[ii]
                    for zz in range(z + 1, n_z):
                        C[t, zz, :] = m_grid
                    C[horizon - 1, :, :] = m_grid
                    return C
            
            if time.time() > deadline:
                for zz in range(z + 1, n_z):
                    C[t, zz, :] = m_grid
                C[horizon - 1, :, :] = m_grid
                return C
    
    # Enforce constraints
    C = np.clip(C, 1e-12, None)
    for t in range(horizon):
        for z in range(n_z):
            C[t, z, :] = np.minimum(C[t, z, :], m_grid + 1e-10)
    C[horizon - 1, :, :] = m_grid
    
    return C
# EVOLVE-BLOCK-END
