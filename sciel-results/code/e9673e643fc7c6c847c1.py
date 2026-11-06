"""QAOA angle design: graph-dependent two-layer parameters via vectorized simulation + multi-start optimization."""
import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = 135.0
_START_TIME = None


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _build_cost_diag(n, edges):
    """Build the diagonal of the cost Hamiltonian C for unweighted MaxCut."""
    m = len(edges)
    if m == 0:
        return np.zeros(2**n)
    
    dim = 2**n
    diag = np.zeros(dim, dtype=np.float64)
    
    for (i, j) in edges:
        states = np.arange(dim)
        bit_i = (states >> i) & 1
        bit_j = (states >> j) & 1
        cut_mask = bit_i ^ bit_j
        diag += cut_mask
    
    return diag


def _build_mixer_indices(n):
    """
    Precompute index pairs for the mixer step.
    For each qubit q, the mixer acts on pairs of basis states that differ only in bit q.
    We return a list of (idx0, idx1) pairs for each qubit where idx0 has bit q = 0 and idx1 has bit q = 1.
    """
    dim = 2**n
    all_indices = []
    for q in range(n):
        pairs = []
        for state in range(dim):
            if (state >> q) & 1 == 0:
                idx0 = state
                idx1 = state | (1 << q)
                pairs.append((idx0, idx1))
        all_indices.append(pairs)
    return all_indices


def _expected_cut_fast(n, edges, gamma1, beta1, gamma2, beta2, cost_diag, mixer_indices):
    """
    Compute expected cut for 2-layer QAOA using vectorized operations.
    """
    dim = 2**n
    
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Apply cost layer 1
    psi *= np.exp(-1j * gamma1 * cost_diag)
    
    # Apply mixer 1
    cb1 = np.cos(beta1)
    sb1 = np.sin(beta1)
    for q in range(n):
        pairs = mixer_indices[q]
        idx0 = np.array([p[0] for p in pairs])
        idx1 = np.array([p[1] for p in pairs])
        old0 = psi[idx0].copy()
        old1 = psi[idx1].copy()
        psi[idx0] = cb1 * old0 - 1j * sb1 * old1
        psi[idx1] = -1j * sb1 * old0 + cb1 * old1
    
    # Apply cost layer 2
    psi *= np.exp(-1j * gamma2 * cost_diag)
    
    # Apply mixer 2
    cb2 = np.cos(beta2)
    sb2 = np.sin(beta2)
    for q in range(n):
        pairs = mixer_indices[q]
        idx0 = np.array([p[0] for p in pairs])
        idx1 = np.array([p[1] for p in pairs])
        old0 = psi[idx0].copy()
        old1 = psi[idx1].copy()
        psi[idx0] = cb2 * old0 - 1j * sb2 * old1
        psi[idx1] = -1j * sb2 * old0 + cb2 * old1
    
    probs = np.abs(psi)**2
    return np.dot(probs, cost_diag)


def _exact_maxcut(n, edges):
    """Exact MaxCut via brute force for small n."""
    if not edges:
        return 0.0
    
    best = 0
    for mask in range(2**n):
        cut = 0
        for (i, j) in edges:
            if ((mask >> i) & 1) != ((mask >> j) & 1):
                cut += 1
        if cut > best:
            best = cut
    return float(best)


def _optimize_for_graph(n, edges):
    """Find best QAOA angles via multi-start Nelder-Mead with vectorized simulation."""
    opt_cut = _exact_maxcut(n, edges)
    if opt_cut == 0:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    cost_diag = _build_cost_diag(n, edges)
    mixer_indices = _build_mixer_indices(n)
    
    def ratio(params):
        g1, b1, g2, b2 = params
        exp_cut = _expected_cut_fast(n, edges, g1, b1, g2, b2, cost_diag, mixer_indices)
        return exp_cut / opt_cut
    
    def objective(params):
        return -ratio(params)
    
    # Multi-start points: hand-crafted + random
    starts = [
        [0.35, 0.25, 0.35, 0.25],
        [0.5, 0.5, 0.4, 0.3],
        [0.2, 0.3, 0.6, 0.4],
        [0.8, 0.2, 0.3, 0.6],
        [0.1, 0.8, 0.7, 0.1],
        [0.9, 0.4, 0.2, 0.7],
        [0.4, 0.6, 0.8, 0.2],
        [0.6, 0.1, 0.5, 0.5],
    ]
    
    # Add random starts
    rng = np.random.RandomState(42)
    for _ in range(4):
        start = rng.uniform(0.05, np.pi - 0.05, size=4)
        starts.append(start.tolist())
    
    best_score = -np.inf
    best_params = None
    
    for start in starts:
        if _remaining() < 8.0:
            break
        
        try:
            result = minimize(
                objective,
                start,
                method='Nelder-Mead',
                options={
                    'maxiter': 1500,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                }
            )
            score = -result.fun
            if score > best_score:
                best_score = score
                best_params = result.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    # Final high-precision polish
    if _remaining() > 6.0:
        try:
            result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 2500,
                    'xatol': 1e-13,
                    'fatol': 1e-15,
                    'adaptive': True,
                }
            )
            score = -result.fun
            if score > best_score:
                best_score = score
                best_params = result.x.copy()
        except Exception:
            pass
    
    return np.array([[best_params[0], best_params[1]],
                     [best_params[2], best_params[3]]])


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of qubits
        edges: list of (i, j) edges
        p: number of layers (should be 2)
    
    Returns:
        np.array of shape (p, 2), rows (gamma, beta) in radians.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    if p != 2:
        return np.tile([0.35, 0.25], (p, 1))
    
    if n > 12:
        return np.array([[0.5, 0.5], [0.4, 0.3]])
    
    try:
        result = _optimize_for_graph(n, edges)
    except Exception:
        result = np.array([[0.35, 0.25], [0.35, 0.25]])
    
    return result
