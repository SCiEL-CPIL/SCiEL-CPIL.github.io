"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

def _build_cost_vector(n, edges):
    """
    Precompute the cut value for all 2^n states.
    Returns a vector E of length 2^n where E[s] is the number of cut edges for state s.
    """
    num_states = 2 ** n
    states = np.arange(num_states)
    E = np.zeros(num_states, dtype=np.float64)
    
    for u, v in edges:
        bu = (states >> u) & 1
        bv = (states >> v) & 1
        E += (bu != bv)
        
    return E

def _evaluate_angles(params, E, n, states):
    """
    Compute the expected cut value for given angles (g1, b1, g2, b2).
    Optimized state vector simulation.
    """
    g1, b1, g2, b2 = params
    num_states = 2 ** n
    
    # Start in |+>^n
    c = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    # Helper to apply mixing in-place
    def apply_mixing(state, beta):
        c_val = np.cos(beta)
        s_val = np.sin(beta)
        for k in range(n):
            mask = 1 << k
            mask_zero = ((states >> k) & 1) == 0
            idx_zero = np.where(mask_zero)[0]
            idx_one = idx_zero ^ mask
            
            v_zero = state[idx_zero]
            v_one = state[idx_one]
            
            state[idx_zero] = c_val * v_zero - 1j * s_val * v_one
            state[idx_one] = c_val * v_one - 1j * s_val * v_zero
        return state

    # Layer 1
    c *= np.exp(-1j * g1 * E)
    c = apply_mixing(c, b1)
    
    # Layer 2
    c *= np.exp(-1j * g2 * E)
    c = apply_mixing(c, b2)
    
    # Expectation value
    probs = np.abs(c) ** 2
    val = np.sum(probs * E)
    return val

def angles(n, edges, p):
    E = _build_cost_vector(n, edges)
    num_states = 2 ** n
    states = np.arange(num_states)
    
    # Objective function for scipy
    def objective(params):
        return -_evaluate_angles(params, E, n, states)
    
    bounds = [(0, 2 * np.pi)] * 4
    
    # Use multiple random restarts with differential evolution
    best_val = -np.inf
    best_params = None
    
    # Try several different random seeds for DE to explore different regions
    for seed in [42, 123, 456, 789, 101]:
        try:
            res_de = differential_evolution(objective, bounds, 
                                            maxiter=80, 
                                            popsize=12,
                                            tol=1e-5,
                                            seed=seed,
                                            polish=False)
            
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_params = res_de.x.copy()
        except Exception:
            continue
    
    # If all DE attempts failed, use random initialization
    if best_params is None:
        best_params = np.random.uniform(0, 2*np.pi, 4)
    
    # Local refinement with Nelder-Mead from the best DE result
    try:
        res_nm = minimize(objective, best_params, method='Nelder-Mead', 
                          options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 3000, 'maxfev': 8000})
        
        if res_nm.fun < best_val:
            best_params = res_nm.x
    except Exception:
        pass
    
    # Also try local refinement from a few other good candidates
    for seed in [42, 123]:
        try:
            res_de = differential_evolution(objective, bounds, 
                                            maxiter=50, 
                                            popsize=10,
                                            tol=1e-5,
                                            seed=seed,
                                            polish=False)
            
            res_nm2 = minimize(objective, res_de.x, method='Nelder-Mead', 
                               options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 2000, 'maxfev': 5000})
            
            if res_nm2.fun < best_val:
                best_val = res_nm2.fun
                best_params = res_nm2.x
        except Exception:
            continue
    
    opt_angles = best_params
    
    # Reshape to (p, 2)
    out = np.array([
        [opt_angles[0], opt_angles[1]],
        [opt_angles[2], opt_angles[3]]
    ])
    
    return out

# EVOLVE-BLOCK-END
