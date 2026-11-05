"""QAOA angle design for MaxCut with p=2 layers using graph-aware lookup and refinement."""
import numpy as np
from scipy.optimize import minimize
import time

# Deadline guard: 150s wall-clock kill, use 140s as safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def _remaining_time():
    return _DEADLINE - time.time()

def _build_adjacency(n, edges):
    """Build adjacency matrix from edge list."""
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj

def _exact_maxcut(n, edges):
    """Compute exact MaxCut via brute force for small n (n <= 10)."""
    m = len(edges)
    if m == 0:
        return 0.0
    best = 0
    # For n=10, 2^10 = 1024 subsets, feasible
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            if (mask >> u) & 1 != (mask >> v) & 1:
                cut += 1
        if cut > best:
            best = cut
    return float(best)

def _precompute_operator_data(n):
    """
    Precompute the eigen-decomposition of the mixer Hamiltonian B = sum_j X_j.
    This allows fast application of exp(-i beta B) via:
    U = V @ diag(exp(-i beta w)) @ V^H
    where B = V @ diag(w) @ V^H
    """
    dim = 1 << n
    B = np.zeros((dim, dim), dtype=np.complex128)
    for j in range(n):
        for x in range(dim):
            y = x ^ (1 << j)
            B[x, y] += 1.0
    
    # B is Hermitian
    w, v = np.linalg.eigh(B)
    return {'w': w, 'v': v, 'dim': dim}

# Cache for operator data
_operator_cache = {}

def _get_operator_data(n):
    if n not in _operator_cache:
        _operator_cache[n] = _precompute_operator_data(n)
    return _operator_cache[n]

def _precompute_graph_data(n, edges):
    """
    Precompute cut values for all 2^n states using vectorized operations.
    Returns cut_vals array of length 2^n.
    """
    dim = 1 << n
    states = np.arange(dim, dtype=np.int32)
    cut_vals = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        mask_i = 1 << i
        mask_j = 1 << j
        bits_i = (states & mask_i) != 0
        bits_j = (states & mask_j) != 0
        cut_vals += (bits_i ^ bits_j).astype(np.float64)
    return cut_vals

# Cache for graph-specific data
_graph_cache = {}

def _get_graph_data(n, edges):
    key = (n, tuple(sorted(edges)))
    if key in _graph_cache:
        return _graph_cache[key]
    
    cut_vals = _precompute_graph_data(n, edges)
    exact_opt = float(np.max(cut_vals))
    
    data = {
        'cut_vals': cut_vals,
        'exact_opt': exact_opt
    }
    _graph_cache[key] = data
    return data

def _expected_cut_fast(n, edges, gammas, betas, op_data, cut_vals):
    """
    Compute expected cut for QAOA with given angles using precomputed operator data.
    C = sum_{(u,v) in edges} (1 - Z_u Z_v)/2, so <C> = m/2 - 0.5 * sum <Z_u Z_v>
    Since C is diagonal in the computational basis, exp(-i gamma C) is just a phase rotation.
    """
    m = len(edges)
    if m == 0:
        return 0.0
    
    dim = op_data['dim']
    w = op_data['w']
    v = op_data['v']
    
    # Initialize state: |+>^n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    p = len(gammas)
    for i in range(p):
        gamma = gammas[i]
        beta = betas[i]
        
        # Apply cost unitary: exp(-i gamma C)
        # C is diagonal, so this is element-wise multiplication
        if abs(gamma) > 1e-10:
            state = state * np.exp(-1j * gamma * cut_vals)
        
        # Apply mixer unitary: exp(-i beta B)
        # U = V @ diag(exp(-i beta w)) @ V^H
        if abs(beta) > 1e-10:
            # state' = V ( exp(-i beta w) * (V^H state) )
            c = v.conj().T @ state
            c = c * np.exp(-1j * beta * w)
            state = v @ c
    
    # Compute expected cut: <C> = sum_k |psi_k|^2 * cuts_k
    prob = np.abs(state) ** 2
    expected = np.dot(prob, cut_vals)
    return expected

def _objective(params, n, edges, exact_opt, op_data, cut_vals):
    """Objective to minimize: negative of the ratio."""
    if _remaining_time() < 2.0:
        return 1.0  # Large penalty if out of time
    
    gammas = params[:2]
    betas = params[2:]
    expected = _expected_cut_fast(n, edges, gammas, betas, op_data, cut_vals)
    if exact_opt == 0:
        return 0.0
    return -expected / exact_opt

def _optimize_angles_for_graph(n, edges):
    """Optimize angles for a single graph."""
    graph_data = _get_graph_data(n, edges)
    cut_vals = graph_data['cut_vals']
    exact_opt = graph_data['exact_opt']
    
    if exact_opt == 0:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    op_data = _get_operator_data(n)
    
    best_params = np.array([0.35, 0.35, 0.25, 0.25])
    best_val = _objective(best_params, n, edges, exact_opt, op_data, cut_vals)
    
    # Try multiple initial points
    init_points = [
        np.array([0.35, 0.35, 0.25, 0.25]),
        np.array([0.5, 0.5, 0.3, 0.3]),
        np.array([0.2, 0.2, 0.1, 0.1]),
        np.array([0.7, 0.7, 0.5, 0.5]),
        np.array([0.1, 0.4, 0.2, 0.1]),
        np.array([0.4, 0.1, 0.1, 0.4]),
        np.array([0.3, 0.3, 0.3, 0.3]),
        np.array([0.5, 0.2, 0.2, 0.5]),
        np.array([0.6, 0.2, 0.5, 0.3]),
        np.array([0.4, 0.3, 0.4, 0.3]),
    ]
    
    for init in init_points:
        if _remaining_time() < 5.0:
            break
        try:
            res = minimize(_objective, init, args=(n, edges, exact_opt, op_data, cut_vals),
                          method='Nelder-Mead',
                          options={'maxiter': 150, 'xatol': 1e-5, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # If time permits, try a second round with more iterations from best point
    if _remaining_time() > 15.0:
        try:
            res2 = minimize(_objective, best_params, args=(n, edges, exact_opt, op_data, cut_vals),
                          method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-7})
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # If time permits, try L-BFGS-B polish pass
    if _remaining_time() > 10.0:
        try:
            res3 = minimize(_objective, best_params, args=(n, edges, exact_opt, op_data, cut_vals),
                          method='L-BFGS-B',
                          options={'maxiter': 100, 'gtol': 1e-10, 'ftol': 1e-10})
            if res3.fun < best_val:
                best_val = res3.fun
                best_params = res3.x.copy()
        except Exception:
            pass
    
    return np.array([[best_params[0], best_params[2]],
                     [best_params[1], best_params[3]]])

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    Returns array of shape (p, 2) with rows (gamma, beta) in radians.
    """
    if p != 2:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Check time budget
    if _remaining_time() < 10.0:
        # Not enough time for optimization, return heuristic
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    try:
        return _optimize_angles_for_graph(n, edges)
    except Exception:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
