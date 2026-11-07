"""Rewritten program using rational function approximation for consumption policy."""
import numpy as np
from scipy.optimize import minimize_scalar, brentq

def solve_consumption(model, resource_grid, budget):
    """
    Solve the consumption-saving problem using backward induction with
    rational function approximation for the value function / policy.
    
    Returns consumption[t, income_state, resource_index] with shape (6, 2, 101).
    """
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)  # 2
    n_grid = len(resource_grid)  # 101
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    # For the last period (t = horizon-1 = 5), the agent consumes all resources
    # because there's no future. So c[5, z, i] = resource_grid[i]
    consumption = np.zeros((horizon, n_states, n_grid))
    consumption[horizon - 1, :, :] = resource_grid[np.newaxis, :, np.newaxis]
    
    # Backward induction for t = horizon-2 down to 0
    # We'll use a parametric representation of the continuation value function
    # V_t(m, z) as a smooth function of m.
    
    # For each state z at time t, we approximate V_t(m, z) using a log-linear or
    # power-law form. Given CRRA utility, the optimal policy is often well-approximated
    # by c = phi(z) * m for some fraction phi(z) in (0, 1).
    
    # Let's use the approach: for each (t, z), find the optimal consumption fraction
    # f(t, z) such that c = f(t, z) * m is optimal.
    
    # Actually, let's do proper backward induction but with efficient optimization.
    
    # State value functions: V[t, z](m) for t from horizon-2 down to 0
    # At t = horizon-1: V[horizon-1, z](m) = utility(m) (consume everything)
    
    # We'll represent V as a function that can be evaluated at any m.
    # Use a dense grid for V and interpolate, but optimize c efficiently.
    
    # Dense grid for m to represent V
    m_dense = np.geomspace(0.001, 40, 2000)
    
    # V[t, z, j] for t in range(horizon-1), z in range(n_states), j in range(len(m_dense))
    V = np.zeros((horizon, n_states, len(m_dense)))
    
    # Base case: t = horizon-1
    V[horizon-1, :, :] = utility(m_dense)[np.newaxis, np.newaxis, :]
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        for z in range(n_states):
            # For each m in m_dense, find optimal c that maximizes:
            # utility(c) + beta * sum_{z'} transition[z, z'] * V[t+1, z'](R*(m-c) + income[z'])
            
            # We'll use the fact that for each m, the optimal c is in [0, m]
            # and the objective is concave in c (since utility is concave and V is concave).
            
            # For efficiency, we'll use a coarse-to-fine approach:
            # First, use a parametric guess c = f * m, then refine.
            
            # Let's define the objective function for a given m
            def neg_obj(c, m_val):
                c = max(c, 1e-12)
                m_val = max(m_val, 0)
                if c > m_val + 1e-10:
                    return 0  # invalid, but shouldn't happen in our search
                m_next = R * (m_val - c) + income
                # m_next is an array of shape (n_states,) for each z'
                # Expected continuation value
                cont_val = np.sum(transition[z] * V[t+1, :, :])
                # Wait, V[t+1, z'](m_next[z']) for each z'
                # m_next[z'] = R*(m_val - c) + income[z']
                # We need to interpolate V[t+1, z'] at m_next[z']
                
                vals = np.zeros(n_states)
                for zp in range(n_states):
                    m_zp = R * (m_val - c) + income[zp]
                    m_zp = max(m_zp, 0.001)  # floor
                    # Interpolate V[t+1, zp] at m_zp
                    if m_zp <= m_dense[0]:
                        vals[zp] = V[t+1, zp, 0]
                    elif m_zp >= m_dense[-1]:
                        vals[zp] = V[t+1, zp, -1]
                    else:
                        vals[zp] = np.interp(m_zp, m_dense, V[t+1, zp])
                ev = np.dot(transition[z], vals)
                return -(utility(c) + beta * ev)
            
            # For each m in m_dense, find optimal c
            # Use a vectorized approach with coarse search first
            
            # Coarse search: c as fraction of m
            fractions = np.linspace(0.01, 1.0, 50)
            c_candidates = m_dense[:, np.newaxis] * fractions[np.newaxis, :]
            c_candidates = np.clip(c_candidates, 1e-12, m_dense[:, np.newaxis])
            
            # Evaluate objective on coarse grid
            # For each (m, c) pair, compute the objective
            obj_grid = np.zeros((len(m_dense), len(fractions)))
            for i, m_val in enumerate(m_dense):
                for j, frac in enumerate(fractions):
                    c_val = m_val * frac
                    if c_val > m_val + 1e-10:
                        obj_grid[i, j] = 0
                    else:
                        obj_grid[i, j] = neg_obj(c_val, m_val)
            
            # Find best fraction for each m
            best_j = np.argmin(obj_grid, axis=1)
            best_c_coarse = m_dense * fractions[best_j]
            
            # Now refine with local optimization around the best coarse solution
            for i, m_val in enumerate(m_dense):
                c0 = best_c_coarse[i]
                # Local search in [max(0, c0 - 0.05*m_val), min(m_val, c0 + 0.05*m_val)]
                lower = max(1e-12, c0 - 0.05 * m_val)
                upper = min(m_val, c0 + 0.05 * m_val)
                
                if lower >= upper:
                    V[t, z, i] = -neg_obj(c0, m_val)
                    continue
                
                # Use minimize_scalar
                res = minimize_scalar(lambda c: neg_obj(c, m_val), 
                                      bounds=(lower, upper), 
                                      method='bounded',
                                      options={'xatol': 1e-8})
                c_opt = res.x
                V[t, z, i] = -res.fun
            
            # Also store the optimal consumption for the original grid
            for idx, m_val in enumerate(resource_grid):
                # Find optimal c for this m_val
                # Use the coarse-to-fine approach
                c_candidates_local = m_val * fractions
                c_candidates_local = np.clip(c_candidates_local, 1e-12, m_val)
                
                obj_local = np.array([neg_obj(c, m_val) for c in c_candidates_local])
                best_j_local = np.argmin(obj_local)
                c0_local = c_candidates_local[best_j_local]
                
                lower = max(1e-12, c0_local - 0.05 * m_val)
                upper = min(m_val, c0_local + 0.05 * m_val)
                
                if lower >= upper:
                    c_opt = c0_local
                else:
                    res = minimize_scalar(lambda c: neg_obj(c, m_val), 
                                          bounds=(lower, upper), 
                                          method='bounded',
                                          options={'xatol': 1e-8})
                    c_opt = res.x
                
                consumption[t, z, idx] = c_opt
    
    # Ensure c[5] equals the grid
    consumption[horizon - 1, :, :] = resource_grid[np.newaxis, :, np.newaxis]
    
    # Enforce constraints: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_states):
            consumption[t, z, :] = np.clip(consumption[t, z, :], 1e-12, resource_grid + 1e-10)
    
    return consumption
