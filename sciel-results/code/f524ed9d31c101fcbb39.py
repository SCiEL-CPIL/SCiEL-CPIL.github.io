"""Improved program: vectorized mixer for speed, enabling more optimization iterations."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Cache for optimal angles per graph
_angle_cache = {}
_T0 = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 15.0

def _get_remaining_time():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _T0)

def _build_c_diagonal(n, edges):
    """
    Compute the diagonal values of the MaxCut Hamiltonian C.
    C(b) = number of edges crossing the cut defined by bitstring b.
    Vectorized computation.
    """
    size = 2**n
    masks = np.arange(size)
    diag = np.zeros(size)
    for u, v in edges:
        # Extract bits u and v for all masks
        bit_u = (masks >> u) & 1
        bit_v = (masks >> v) & 1
        # Add 1 where bits differ
        diag += (bit_u != bit_v).astype(float)
    return diag

def _apply_mixer_vectorized(state, beta, n):
    """
    Apply exp(-i beta sum X) to state using vectorized operations.
    exp(-i beta sum X) = prod_j (cos(beta) I_j - i sin(beta) X_j)
    This is a tensor product of identical single-qubit gates.
    """
    c = np.cos(beta)
    s = -1j * np.sin(beta)
    
    # Reshape state to tensor
    state_tensor = state.reshape([2] * n)
    
    # For each qubit axis, apply the gate [[c, s], [s, c]]
    # We can do this efficiently by swapping and combining
    for j in range(n):
        # Move axis j to front
        state_tensor = np.moveaxis(state_tensor, j, 0)
        orig_shape = state_tensor.shape
        state_tensor = state_tensor.reshape(-1, 2)
        
        # Apply gate: new[0] = c*old[0] + s*old[1], new[1] = s*old[0] + c*old[1]
        old0 = state_tensor[:, 0]
        old1 = state_tensor[:, 1]
        new0 = c * old0 + s * old1
        new1 = s * old0 + c * old1
        
        state_tensor = np.column_stack((new0, new1))
        state_tensor = state_tensor.reshape(orig_shape)
        state_tensor = np.moveaxis(state_tensor, 0, j)
    
    return state_tensor.reshape(-1)

def _apply_cost(state, gamma, c_diag):
    """
    Apply exp(-i gamma C).
    C is diagonal with values c_diag.
    """
    phase = np.exp(-1j * gamma * c_diag)
    return state * phase

def _expected_cut_value(n, edges, gammas, betas, c_diag=None):
    """
    Compute expected cut for given parameters.
    """
    if c_diag is None:
        c_diag = _build_c_diagonal(n, edges)
    
    p = len(gammas)
    # Initial state |+>^n
    state = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
    
    for k in range(p):
        # Apply Cost
        state = _apply_cost(state, gammas[k], c_diag)
        # Apply Mixer
        state = _apply_mixer_vectorized(state, betas[k], n)
        
    # Compute expectation <psi| C |psi>
    prob = np.abs(state)**2
    return float(np.dot(prob, c_diag))

def _optimize_angles(n, edges, p):
    """
    Optimize angles for given graph using multiple restarts and time-aware iteration.
    """
    # Precompute cost diagonal
    c_diag = _build_c_diagonal(n, edges)
    
    best_val = -np.inf
    best_params = None
    
    # Objective function to minimize (negative expected cut)
    def obj(params):
        gammas = params[:p]
        betas = params[p:]
        val = _expected_cut_value(n, edges, gammas, betas, c_diag)
        return -val
    
    # Define starting points - more diverse starts for better coverage
    starts = []
    
    # Standard heuristic starts
    base_starts = [
        [0.35, 0.25],  # Seed values
        [0.5, 0.2],
        [1.0, 0.5],
        [np.pi/4, np.pi/8],
        [0.7, 0.3],
        [0.2, 0.4],
        [np.pi/3, np.pi/6],
        [0.8, 0.15],
    ]
    
    for bs in base_starts:
        starts.append(bs * p)
    
    # Add some random starts
    np.random.seed(42)
    for _ in range(4):
        rand_start = np.random.uniform(0, np.pi, 2 * p)
        starts.append(rand_start)
    
    # Sort starts by heuristic quality (try good ones first)
    # Evaluate each start and sort by objective value
    start_objs = []
    for start in starts:
        try:
            obj_val = obj(start)
            start_objs.append((obj_val, start))
        except:
            start_objs.append((np.inf, start))
    
    start_objs.sort(key=lambda x: x[0])
    
    # Optimize from best starts first
    for obj_val, start in start_objs:
        # Check time
        if _get_remaining_time() < 3.0:
            break
            
        try:
            # Use Nelder-Mead with reasonable settings
            res = minimize(obj, start, method='Nelder-Mead', 
                           options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
                
                # If we found a good solution, try to refine with tighter tolerances
                if _get_remaining_time() > 5.0 and best_val < -3.0:
                    try:
                        res2 = minimize(obj, best_params, method='Nelder-Mead', 
                                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
                        if res2.fun < best_val:
                            best_val = res2.fun
                            best_params = res2.x.copy()
                    except:
                        pass
        except Exception:
            continue
    
    if best_params is None:
        # Fallback to seed
        return np.tile([0.35, 0.25], (p, 1))
        
    # Extract gammas and betas
    gammas = best_params[:p]
    betas = best_params[p:]
    
    # Format output: (p, 2) array, rows (gamma, beta)
    angles = np.column_stack((gammas, betas))
    return angles

def angles(n, edges, p):
    # Create a cache key
    edges_tuple = tuple(sorted((u, v) if u < v else (v, u) for u, v in edges))
    key = (n, edges_tuple)
    
    if key in _angle_cache:
        return _angle_cache[key]
        
    # Check time
    if _get_remaining_time() < 2.0:
        # Not enough time to optimize, return default
        res = np.tile([0.35, 0.25], (p, 1))
        return res
        
    res = _optimize_angles(n, edges, p)
    
    # Store in cache
    _angle_cache[key] = res
    
    return res
# EVOLVE-BLOCK-END
