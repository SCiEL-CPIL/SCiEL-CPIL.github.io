"""QAOA angle design for MaxCut with p=2 layers using graph-aware optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 150.0  # seconds, with safety margin
_SAFETY_MARGIN = 10.0  # seconds before deadline to stop

def _remaining_time():
    """Return remaining time before deadline with safety margin."""
    return _DEADLINE - (time.time() - _START_TIME) - _SAFETY_MARGIN

def _build_adjacency(n, edges):
    """Build adjacency matrix from edge list."""
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj

def _exact_maxcut(n, edges):
    """Compute exact MaxCut value by exhaustive search."""
    m = len(edges)
    if m == 0:
        return 0.0
    best_cut = 0
    # For small n (<=10), exhaustive search is feasible
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best_cut:
            best_cut = cut
    return float(best_cut)

def _precompute_operator_data(n, edges):
    """
    Precompute data needed for fast QAOA expected cut evaluation.
    
    For p=2 QAOA on MaxCut, the expected cut value can be computed
    using the structure of the Hamiltonians.
    
    C = sum_{(u,v) in E} (1 - Z_u Z_v) / 2
    B = sum_j X_j
    
    The state after p=2 layers:
    |psi> = exp(-i beta_1 B) exp(-i gamma_1 C) exp(-i beta_2 B) exp(-i gamma_2 C) |+>^n
    
    For the expected cut, we need <psi|C|psi>.
    """
    adj = _build_adjacency(n, edges)
    m = len(edges)
    
    # For efficient computation, we precompute various terms
    # The key insight is that for small n, we can use the full state vector
    # but for larger n, we need tensor network or other methods.
    
    # For n <= 10, we can use the full Hilbert space (2^n states)
    dim = 1 << n
    
    # Identity
    I = np.eye(dim, dtype=np.complex128)
    
    # Build Z operators (diagonal)
    Z_ops = []
    for j in range(n):
        Z = np.zeros((dim, dim), dtype=np.complex128)
        for s in range(dim):
            val = 1.0 if (s >> j) & 1 == 0 else -1.0
            Z[s, s] = val
        Z_ops.append(Z)
    
    # Build X operators
    X_ops = []
    for j in range(n):
        X = np.zeros((dim, dim), dtype=np.complex128)
        for s in range(dim):
            s2 = s ^ (1 << j)
            X[s, s2] = 1.0
        X_ops.append(X)
    
    # Build C = sum_{(u,v) in E} (1 - Z_u Z_v) / 2
    C = np.zeros((dim, dim), dtype=np.complex128)
    for u, v in edges:
        C += (I - Z_ops[u] @ Z_ops[v]) / 2.0
    
    # Build B = sum_j X_j
    B = np.zeros((dim, dim), dtype=np.complex128)
    for j in range(n):
        B += X_ops[j]
    
    # Initial state |+>^n
    psi0 = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    return {
        'dim': dim,
        'C': C,
        'B': B,
        'psi0': psi0,
        'n': n,
        'm': m
    }

def _simulate_qaoa(op_data, gamma1, beta1, gamma2, beta2):
    """
    Simulate 2-layer QAOA and return expected cut value.
    Order: exp(-i gamma1 C) first, then exp(-i beta1 B), etc.
    Actually the circuit is: for each row (gamma, beta):
    apply exp(-i gamma C), then exp(-i beta B)
    So for p=2: exp(-i beta1 B) exp(-i gamma1 C) exp(-i beta2 B) exp(-i gamma2 C) |+>^n
    Wait, let me re-read: "applies exp(-i gamma C), then exp(-i beta sum X), for each row"
    So row 0: exp(-i gamma0 C) then exp(-i beta0 B)
    Row 1: exp(-i gamma1 C) then exp(-i beta1 B)
    Total: exp(-i beta1 B) exp(-i gamma1 C) exp(-i beta0 B) exp(-i gamma0 C) |+>^n
    """
    dim = op_data['dim']
    C = op_data['C']
    B = op_data['B']
    psi0 = op_data['psi0']
    
    # Start with |+>^n
    psi = psi0.copy()
    
    # Apply layers in reverse order (since operators act on state from right to left)
    # Actually, we apply from rightmost to leftmost when evolving the state
    # State evolution: |psi> = U |psi0> where U = U_p ... U_2 U_1
    # Each layer: exp(-i beta B) exp(-i gamma C)
    # So U = exp(-i beta1 B) exp(-i gamma1 C) exp(-i beta0 B) exp(-i gamma0 C)
    
    # Apply exp(-i gamma0 C)
    # Use matrix exponential for small matrices
    from scipy.linalg import expm
    U = expm(-1j * gamma1 * C)
    psi = U @ psi
    U = expm(-1j * beta1 * B)
    psi = U @ psi
    U = expm(-1j * gamma0 * C)
    psi = U @ psi
    U = expm(-1j * beta0 * B)
    psi = U @ psi
    
    # Expected cut = <psi|C|psi>
    expected_cut = np.real(np.vdot(psi, C @ psi))
    
    return expected_cut

def _expected_cut_fast(op_data, gamma0, beta0, gamma1, beta1):
    """
    Fast computation of expected cut for 2-layer QAOA.
    Uses the fact that for small n, we can compute this efficiently.
    """
    # For n <= 10, use full simulation
    dim = op_data['dim']
    if dim <= 1024:  # n <= 10
        return _simulate_qaoa(op_data, gamma0, beta0, gamma1, beta1)
    else:
        # Fallback: use a simpler approximation
        # For now, return a reasonable estimate
        m = op_data['m']
        # Random state gives m/2 on average
        return m / 2.0

def _objective(params, op_data, opt_cut):
    """Compute the ratio: expected_cut / exact_optimum."""
    gamma0, beta0, gamma1, beta1 = params
    exp_cut = _expected_cut_fast(op_data, gamma0, beta0, gamma1, beta1)
    if opt_cut == 0:
        return 0.0
    return exp_cut / opt_cut

def _optimize_angles_for_graph(n, edges, p=2):
    """
    Optimize QAOA angles for a given graph.
    Uses multi-start local optimization.
    """
    # Precompute operator data
    op_data = _precompute_operator_data(n, edges)
    opt_cut = _exact_maxcut(n, edges)
    
    if opt_cut == 0:
        return np.zeros((p, 2))
    
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    best_obj = _objective(best_params, op_data, opt_cut)
    
    # Multi-start optimization
    from scipy.optimize import minimize
    
    # Define bounds: angles in [0, pi]
    bounds = [(0, np.pi), (0, np.pi), (0, np.pi), (0, np.pi)]
    
    # Try multiple starting points
    np.random.seed(42)
    num_starts = 20
    
    for start in range(num_starts):
        if _remaining_time() < 5:
            break
            
        if start == 0:
            x0 = best_params.copy()
        else:
            x0 = np.random.uniform(0, np.pi, 4)
        
        try:
            result = minimize(
                lambda x: -_objective(x, op_data, opt_cut),
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            if result.fun < -best_obj:
                best_obj = -result.fun
                best_params = result.x
        except Exception:
            continue
    
    # Refine with Nelder-Mead
    try:
        result = minimize(
            lambda x: -_objective(x, op_data, opt_cut),
            best_params,
            method='Nelder-Mead',
            bounds=None,
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if result.fun < -best_obj:
            best_obj = -result.fun
            best_params = result.x
    except Exception:
        pass
    
    # Clamp to valid range
    best_params = np.clip(best_params, 0, np.pi)
    
    # Reshape to (p, 2)
    angles = best_params.reshape(p, 2)
    
    return angles

# Cache for operator data and optimization results
_op_data_cache = {}
_angles_cache = {}

def angles(n, edges, p):
    """
    Return QAOA angles for the given graph.
    
    Args:
        n: number of qubits
        edges: list of (u, v) edges
        p: number of QAOA layers (should be 2)
    
    Returns:
        Array of shape (p, 2) with rows (gamma, beta) in radians
    """
    # Create a cache key
    edge_set = frozenset(edges)
    cache_key = (n, edge_set, p)
    
    if cache_key in _angles_cache:
        return _angles_cache[cache_key]
    
    # Check time
    if _remaining_time() < 10:
        # Return default angles if not enough time
        result = np.tile([0.35, 0.25], (p, 1))
        _angles_cache[cache_key] = result
        return result
    
    # Optimize angles for this graph
    result = _optimize_angles_for_graph(n, edges, p)
    
    _angles_cache[cache_key] = result
    return result
# EVOLVE-BLOCK-END
