"""QAOA parameter design with exact simulation and L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Cache for exact optima
_optima_cache = {}
# Cache for optimized angles per graph signature
_angles_cache = {}

def _build_edge_list(n, edges):
    """Build edge list from various input formats."""
    if edges is None:
        return []
    if isinstance(edges, np.ndarray):
        if edges.ndim == 2 and edges.shape[1] == 2:
            return [(int(edges[i, 0]), int(edges[i, 1])) for i in range(edges.shape[0])]
        elif edges.ndim == 2 and edges.shape == (n, n):
            return [(i, j) for i in range(n) for j in range(i+1, n) if edges[i, j] != 0]
    if isinstance(edges, list):
        if len(edges) > 0 and isinstance(edges[0], (list, tuple)) and len(edges[0]) == 2:
            return [(int(e[0]), int(e[1])) for e in edges]
    return []

def _exact_optimum(n, edges):
    """Compute exact MaxCut optimum via brute force."""
    key = (n, tuple(sorted(edges)))
    if key in _optima_cache:
        return _optima_cache[key]
    
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    
    best = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in edge_set:
            if (mask >> u & 1) != (mask >> v & 1):
                cut += 1
        if cut > best:
            best = cut
    
    _optima_cache[key] = best
    return best

def _expected_cut_exact(n, edges, gammas, betas):
    """
    Compute expected cut value using exact state vector simulation.
    Uses the fact that C (MaxCut Hamiltonian) is diagonal in Z-basis.
    
    The QAOA circuit: |psi> = product_{l=1}^{p} [exp(-i*beta_l*sum X_i) * exp(-i*gamma_l*C)] |0>^n
    
    Expected cut = <psi| C |psi>
    
    Since C is diagonal: C = sum_{(u,v) in edges} (1 - Z_u Z_v)/2
    <C> = sum_{(u,v) in edges} (1 - <Z_u Z_v>)/2
    """
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    
    if not edge_set:
        return 0.0
    
    # State vector: 2^n complex amplitudes
    dim = 1 << n
    # Start in |+>^n = (1/sqrt(2))^n * sum_z |z>
    # All amplitudes equal: 1/sqrt(2^n)
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute diagonal of C for each basis state
    # C|z> = c_z |z> where c_z = sum_{(u,v) in edges} (1 - (-1)^{z_u z_v})/2
    # c_z = number of edges cut by partition z
    
    diag_C = np.zeros(dim, dtype=np.float64)
    for z in range(dim):
        cut = 0
        for u, v in edge_set:
            if ((z >> u) & 1) != ((z >> v) & 1):
                cut += 1
        diag_C[z] = cut
    
    # Apply QAOA layers
    for l in range(len(gammas)):
        # Apply exp(-i * gamma_l * C)
        # This multiplies each amplitude by exp(-i * gamma_l * diag_C[z])
        phases = np.exp(-1j * gammas[l] * diag_C)
        psi = psi * phases
        
        # Apply exp(-i * beta_l * sum_i X_i)
        # sum_i X_i is the mixer: it flips all bits simultaneously
        # exp(-i*beta*sum X_i) = cos(beta) * I - i*sin(beta) * X^{\otimes n}
        # X^{\otimes n} |z> = |z_flipped> where z_flipped = z XOR (2^n - 1)
        beta = betas[l]
        c = np.cos(beta)
        s = np.sin(beta)
        
        # X^{\otimes n} reverses the state vector (bit reversal of all n bits)
        # Actually, X|0> = |1>, X|1> = |0>, so X^{\otimes n} maps |z> to |~z & (2^n-1)>
        # In terms of index: index z maps to index (dim-1-z)
        psi_flipped = psi[::-1].copy()
        psi = c * psi - 1j * s * psi_flipped
    
    # Compute <C> = sum_z |psi_z|^2 * diag_C[z]
    probs = np.abs(psi) ** 2
    expected_cut = np.dot(probs, diag_C)
    
    return float(expected_cut)

def _optimize_for_graph(n, edges, p, deadline):
    """Optimize QAOA angles for a specific graph."""
    key = (n, tuple(sorted(edges)), p)
    if key in _angles_cache:
        return _angles_cache[key]
    
    if time.time() > deadline:
        # Return default angles
        return np.tile([0.35, 0.25], (p, 1))
    
    opt = _exact_optimum(n, edges)
    if opt == 0:
        return np.tile([0.0, 0.0], (p, 1))
    
    def objective(params):
        if time.time() > deadline:
            return 0.0
        gammas = params[:p]
        betas = params[p:]
        ec = _expected_cut_exact(n, edges, gammas, betas)
        return -ec / opt  # Negate for minimization
    
    # Multi-start L-BFGS-B
    best_params = None
    best_val = np.inf
    
    # Generate multiple starting points
    n_starts = 8
    rng = np.random.RandomState(42)
    
    starts = []
    # Random starts in reasonable range
    for i in range(n_starts):
        g = rng.uniform(0.0, np.pi, size=p)
        b = rng.uniform(0.0, np.pi, size=p)
        starts.append(np.concatenate([g, b]))
    
    # Add heuristic starts
    starts.append(np.concatenate([np.full(p, 0.5), np.full(p, 0.3)]))
    starts.append(np.concatenate([np.full(p, 1.0), np.full(p, 0.5)]))
    
    for start in starts:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(objective, start, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            continue
    
    if best_params is None:
        # Fallback
        best_params = np.concatenate([np.full(p, 0.35), np.full(p, 0.25)])
    
    angles_result = np.column_stack([best_params[:p], best_params[p:]])
    _angles_cache[key] = angles_result
    return angles_result

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of qubits
        edges: list of edges or adjacency matrix
        p: number of QAOA layers (always 2 per task)
    
    Returns:
        numpy array of shape (p, 2) with rows (gamma, beta) in radians
    """
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    edge_list = _build_edge_list(n, edges)
    
    # Check cache
    key = (n, tuple(sorted(edge_list)), p)
    if key in _angles_cache:
        return _angles_cache[key]
    
    # Optimize
    result = _optimize_for_graph(n, edge_list, p, deadline)
    
    return result
# EVOLVE-BLOCK-END
