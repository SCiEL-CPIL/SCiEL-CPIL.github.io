import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

# Global cache for precomputed graph data
_graph_cache = {}

def _simulate_qaoa_fast(n, edges, params, cut_sizes, flip_indices):
    """
    Simulate QAOA for given graph and parameters.
    params: array of shape (p, 2) with (gamma, beta).
    Returns expected cut size.
    """
    p = len(params)
    N = 1 << n
    state = np.ones(N, dtype=complex) / np.sqrt(N)
    
    for gamma, beta in params:
        # Apply Cost
        state *= np.exp(-1j * gamma * cut_sizes)
        
        # Apply Mixer
        c = np.cos(beta)
        s = -1j * np.sin(beta)
        
        for i in range(n):
            # Use precomputed flip indices
            state = c * state + s * state[flip_indices[i]]
            
    probs = np.abs(state)**2
    return np.sum(probs * cut_sizes)

def _expected_cut_from_params(params_flat, n, cut_sizes, flip_indices):
    """
    Compute expected cut from flat parameter array [gamma1, beta1, gamma2, beta2].
    """
    p = len(params_flat) // 2
    params = params_flat.reshape(p, 2)
    return _simulate_qaoa_fast(n, None, params, cut_sizes, flip_indices)

def angles(n, edges, p):
    n = int(n)
    # Normalize edges to list of tuples to ensure hashability and consistent sorting
    if isinstance(edges, np.ndarray):
        edges = [tuple(e) for e in edges]
    else:
        edges = [tuple(e) for e in edges]

    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    # Get or compute graph data
    edge_key = (n, tuple(sorted(edges)))
    if edge_key in _graph_cache:
        data = _graph_cache[edge_key]
    else:
        # Compute cut sizes
        N = 1 << n
        cut_sizes = np.zeros(N, dtype=float)
        states = np.arange(N, dtype=np.int64)
        for u, v in edges:
            bits_u = (states >> u) & 1
            bits_v = (states >> v) & 1
            cut_sizes += (bits_u != bits_v).astype(float)
        
        max_cut = float(np.max(cut_sizes))
        flip_indices = [np.arange(N) ^ (1 << i) for i in range(n)]
        
        data = {
            'cut_sizes': cut_sizes,
            'flip_indices': flip_indices,
            'max_cut': max_cut,
            'n': n
        }
        _graph_cache[edge_key] = data
    
    cut_sizes = data['cut_sizes']
    flip_indices = data['flip_indices']
    max_cut = float(data['max_cut'])
    
    if max_cut == 0.0:
        return np.zeros((p, 2))
    
    # Define objective function
    def local_obj(params):
        exp_cut = _expected_cut_from_params(params, n, cut_sizes, flip_indices)
        return float(-exp_cut / max_cut)
    
    # Physics-informed initial guesses based on asymptotic QAOA for MaxCut
    initial_guesses = [
        [0.5, 0.5, 0.5, 0.5],
        [0.7, 0.4, 0.7, 0.4],
        [0.3, 0.3, 0.3, 0.3],
        [0.2, 0.6, 0.2, 0.6],
        [1.0, 0.5, 1.0, 0.5],
        [0.4, 0.2, 0.4, 0.2],
        [0.8, 0.3, 0.8, 0.3],
        [0.1, 0.1, 0.1, 0.1],
        [0.6, 0.5, 0.6, 0.5],
        [0.9, 0.3, 0.9, 0.3],
    ]
    
    bounds = [(0, np.pi)] * (2 * p)
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Local optimization from multiple starting points
    for x0 in initial_guesses:
        if time.time() - start_time > deadline - 30.0:
            break
        try:
            res = minimize(
                local_obj, x0, method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            score = float(-res.fun)
            if score > best_score:
                best_score = score
                best_params = res.x
        except Exception:
            continue
    
    # Phase 2: Global optimization with Differential Evolution if time permits
    if time.time() - start_time < deadline - 20.0:
        try:
            de_result = differential_evolution(
                local_obj, 
                bounds, 
                maxiter=40, 
                popsize=15,
                seed=42,
                polish=True,
                tol=1e-5
            )
            de_score = float(-de_result.fun)
            if de_score > best_score:
                best_score = de_score
                best_params = de_result.x
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if best_params is not None and time.time() - start_time < deadline - 5.0:
        try:
            final_res = minimize(
                local_obj, best_params, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            final_score = float(-final_res.fun)
            if final_score > best_score:
                best_score = final_score
                best_params = final_res.x
        except Exception:
            pass
    
    if best_params is None:
        return np.array([[0.5, 0.5], [0.5, 0.5]])
    
    return np.array(best_params).reshape(p, 2)
