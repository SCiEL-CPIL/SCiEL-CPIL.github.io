"""
Quantum information: fixed-depth QAOA parameter design.

Uses exact combinatorial probability aggregation over all 2^n bitmasks to
compute the expected cut value for given QAOA angles, then optimizes angles
per graph using multi-start local optimization.
"""
import numpy as np
from scipy.optimize import minimize


def _build_edge_list(n, edges):
    """Return list of (u, v) for each edge, with u < v."""
    edge_set = set()
    for e in edges:
        u, v = int(e[0]), int(e[1])
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    return sorted(edge_set)


def _precompute_cut_counts(n, edge_list):
    """Precompute cut count for all 2^n bitmasks."""
    dim = 1 << n
    cut_counts = np.zeros(dim, dtype=np.int32)
    num_edges = len(edge_list)
    if num_edges == 0:
        return cut_counts
    for idx in range(dim):
        count = 0
        for (u, v) in edge_list:
            if ((idx >> u) & 1) != ((idx >> v) & 1):
                count += 1
        cut_counts[idx] = count
    return cut_counts


def _expected_cut_fast(n, edge_list, cut_counts, gammas, betas):
    """
    Compute expected cut using precomputed cut counts and vectorized operations.
    """
    dim = 1 << n
    num_edges = len(edge_list)
    
    # Start in |+>^n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for l in range(len(gammas)):
        gamma = gammas[l]
        beta = betas[l]
        
        # Apply cost gate: phase = exp(-i * gamma * cut_count)
        # C|x> = cut(x)|x>, so exp(-i*gamma*C)|x> = exp(-i*gamma*cut(x))|x>
        phases = np.exp(-1j * gamma * cut_counts.astype(np.float64))
        psi *= phases
        
        # Apply mixer: product of single-qubit rotations
        c = np.cos(beta)
        s = np.sin(beta)
        for q in range(n):
            mask = 1 << q
            # Vectorized application of 2x2 rotation
            # For indices where bit q is 0:
            idx0 = np.array([i for i in range(dim) if (i & mask) == 0])
            idx1 = idx0 | mask
            a0 = psi[idx0].copy()
            a1 = psi[idx1].copy()
            psi[idx0] = c * a0 - 1j * s * a1
            psi[idx1] = -1j * s * a0 + c * a1
    
    # Compute expected cut
    probs = np.abs(psi) ** 2
    total_cut = np.sum(probs * cut_counts.astype(np.float64))
    
    return float(total_cut)


def _exact_optimum(n, edge_list, cut_counts):
    """Compute the exact maximum cut."""
    if len(edge_list) == 0:
        return 0.0
    return float(np.max(cut_counts))


def _optimize_angles(n, edges, p, max_iters=200):
    """Optimize QAOA angles for a given graph."""
    edge_list = _build_edge_list(n, edges)
    cut_counts = _precompute_cut_counts(n, edge_list)
    opt_cut = _exact_optimum(n, edge_list, cut_counts)
    
    if opt_cut == 0:
        return np.zeros((p, 2))
    
    def objective(params):
        gammas = params[:p]
        betas = params[p:2*p]
        expected = _expected_cut_fast(n, edge_list, cut_counts, gammas, betas)
        return -expected / opt_cut
    
    # Try multiple initializations and keep the best
    best_obj = np.inf
    best_params = None
    
    # Deterministic initializations based on known good QAOA angles
    inits = [
        np.array([0.5, 0.3] * p),
        np.array([0.4, 0.2] * p),
        np.array([0.3, 0.4] * p),
        np.array([0.6, 0.1] * p),
        np.array([0.2, 0.5] * p),
        np.array([0.7, 0.2] * p),
        np.array([0.3, 0.3] * p),
        np.array([0.5, 0.5] * p),
    ]
    
    # Add some random initializations
    rng = np.random.default_rng(42)
    for _ in range(4):
        inits.append(rng.uniform(0.1, 0.8, 2*p))
    
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': max_iters, 'xatol': 1e-5, 'fatol': 1e-7})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Also try Powell method from best Nelder-Mead result
    if best_params is not None:
        try:
            res2 = minimize(objective, best_params, method='Powell',
                           options={'maxiter': 50, 'xtol': 1e-5, 'ftol': 1e-7})
            if res2.fun < best_obj:
                best_obj = res2.fun
                best_params = res2.x
        except Exception:
            pass
    
    if best_params is None:
        best_params = np.array([0.4, 0.3] * p)
    
    gammas = best_params[:p]
    betas = best_params[p:2*p]
    
    return np.column_stack([gammas, betas])


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio
    of the independently simulated expected cut to the exact classical optimum.
    
    Args:
        n: number of nodes
        edges: list of edges (u, v)
        p: number of layers (fixed to 2 in this task)
    
    Returns:
        np.ndarray of shape (p, 2), rows (gamma, beta) in radians.
    """
    if p == 0:
        return np.zeros((0, 2))
    
    result = _optimize_angles(n, edges, p)
    
    return result
