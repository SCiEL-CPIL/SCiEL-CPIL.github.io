# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_EPS = 1e-12
_SLOPES = [-1.0, 0.0, 1.0, 2.0]

# Precompute projection indices for each slope
# P[y, x] -> index y*N + x
# v = x + slope * y
# We need to map each cell to a bin index for counts.
# v_min and v_max depend on slope.
# Slope -1: v = x - y. Range: -9 to 9. Size 19.
# Slope 0: v = x. Range: 0 to 9. Size 10.
# Slope 1: v = x + y. Range: 0 to 18. Size 19.
# Slope 2: v = x + 2y. Range: 0 to 27. Size 28.

def _get_proj_indices(slope):
    n = _N
    indices = np.zeros((n, n), dtype=np.int32)
    v_min = None
    v_max = None
    for y in range(n):
        for x in range(n):
            v = x + slope * y
            if v_min is None:
                v_min = v
                v_max = v
            else:
                v_min = min(v_min, v)
                v_max = max(v_max, v)
    size = int(round(v_max - v_min)) + 1
    for y in range(n):
        for x in range(n):
            v = x + slope * y
            indices[y, x] = int(round(v - v_min))
    return indices, size

PROJ_INDICES = {}
PROJ_SIZES = {}
for s in _SLOPES:
    idx, sz = _get_proj_indices(s)
    PROJ_INDICES[s] = idx
    PROJ_SIZES[s] = sz

def _shannon_entropy_bits(counts):
    total = counts.sum()
    if total <= _EPS:
        return 0.0
    p = counts / total
    p = p[p > _EPS]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))

def _compute_entropies(weights, support):
    """
    weights: array of length len(support), sum 1.
    support: array of flat indices (y*N + x).
    Returns (H_minus1, H_0, H_1, H_2).
    """
    Hs = []
    for s in _SLOPES:
        counts = np.zeros(PROJ_SIZES[s], dtype=np.float64)
        # Get the bin indices for the cells in support
        bins = PROJ_INDICES[s].ravel()[support]
        np.add.at(counts, bins, weights)
        Hs.append(_shannon_entropy_bits(counts))
    return Hs[0], Hs[1], Hs[2], Hs[3]

def _ratio_from_entropies(Hm1, H0, H1, H2):
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return Hm1 / M

def _objective_on_support(theta, support):
    """
    theta: free parameters for softmax.
    Returns negative ratio (for minimization).
    """
    # Softmax to get weights
    z = theta - np.max(theta)
    exp_z = np.exp(z)
    s = exp_z.sum()
    if s < _EPS:
        weights = np.ones(len(support)) / len(support)
    else:
        weights = exp_z / s
        
    Hm1, H0, H1, H2 = _compute_entropies(weights, support)
    r = _ratio_from_entropies(Hm1, H0, H1, H2)
    return -r

def _optimize_support(support, deadline):
    """
    Optimize weights for a given support set.
    Returns (best_ratio, best_P_full).
    """
    K = len(support)
    if K == 0:
        return 0.0, None
    
    # Initial guess: uniform
    theta0 = np.zeros(K)
    
    best_ratio = 0.0
    best_theta = theta0
    
    # Try a few random initializations
    rng = np.random.default_rng(42)
    attempts = 0
    max_attempts = 10 if K < 15 else 3
    
    while time.time() < deadline - 1.0 and attempts < max_attempts:
        attempts += 1
        if attempts == 1:
            theta_start = np.zeros(K)
        else:
            theta_start = rng.standard_normal(K) * 0.5
        
        try:
            res = minimize(
                _objective_on_support,
                theta_start,
                args=(support,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun is not None:
                ratio = -res.fun
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_theta = res.x
        except Exception:
            pass
            
    if best_ratio <= 0:
        return 0.0, None
        
    # Construct full P
    z = best_theta - np.max(best_theta)
    exp_z = np.exp(z)
    s = exp_z.sum()
    weights = exp_z / s
    
    P = np.zeros(_N * _N, dtype=np.float64)
    P[support] = weights
    P = P.reshape(_N, _N)
    
    return best_ratio, P

def _random_support(rng, min_size=5, max_size=20):
    K = rng.integers(min_size, max_size + 1)
    return rng.choice(_N * _N, size=K, replace=False)

def _structured_supports():
    """Generate some structured supports."""
    supports = []
    n = _N
    # Diagonals
    for d in range(-n+1, n):
        cells = []
        for y in range(n):
            x = y + d
            if 0 <= x < n:
                cells.append(y * n + x)
        if len(cells) >= 3:
            supports.append(np.array(cells))
            
    # Anti-diagonals
    for s in range(0, 2*n-1):
        cells = []
        for y in range(n):
            x = s - y
            if 0 <= x < n:
                cells.append(y * n + x)
        if len(cells) >= 3:
            supports.append(np.array(cells))
            
    # Slope -2 lines (x + 2y = c)
    for c in range(0, 2*n):
        cells = []
        for y in range(n):
            x = c - 2*y
            if 0 <= x < n:
                cells.append(y * n + x)
        if len(cells) >= 3:
            supports.append(np.array(cells))
            
    # Slope 2 lines (x - 2y = c) -> x = 2y + c
    for c in range(-2*n, 2*n):
        cells = []
        for y in range(n):
            x = 2*y + c
            if 0 <= x < n:
                cells.append(y * n + x)
        if len(cells) >= 3:
            supports.append(np.array(cells))
            
    # Rows and Cols
    for i in range(n):
        supports.append(np.array([i * n + j for j in range(n)])) # Row i
        supports.append(np.array([j * n + i for j in range(n)])) # Col i
        
    return supports

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0 # 10s margin
    
    best_ratio = 0.0
    best_P = np.ones((_N, _N)) / (_N * _N)
    
    rng = np.random.default_rng(42)
    
    # 1. Try structured supports
    struct_supports = _structured_supports()
    for support in struct_supports:
        if time.time() > deadline - 30.0:
            break
        ratio, P = _optimize_support(support, deadline)
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P
            
    # 2. Try random supports
    # We can do many of these quickly
    num_random = 0
    while time.time() < deadline - 10.0:
        num_random += 1
        # Vary support size
        if num_random % 10 == 0:
            min_sz, max_sz = 5, 10
        elif num_random % 10 == 5:
            min_sz, max_sz = 10, 15
        else:
            min_sz, max_sz = 15, 25
            
        support = _random_support(rng, min_sz, max_sz)
        ratio, P = _optimize_support(support, deadline)
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P
            
        # If we've been going for a while, increase support size or change strategy
        if num_random % 50 == 0:
             # Try a support based on current best P's non-zero elements
             if best_ratio > 0:
                 nz = np.nonzero(best_P.ravel() > 1e-6)[0]
                 if len(nz) > 0:
                     # Perturb this support
                     perturbed = nz.copy()
                     # Replace a few elements
                     for _ in range(3):
                         idx_to_remove = rng.integers(len(perturbed))
                         perturbed = np.delete(perturbed, idx_to_remove)
                         new_elem = rng.integers(_N * _N)
                         if new_elem not in perturbed:
                             perturbed = np.append(perturbed, new_elem)
                     ratio, P = _optimize_support(perturbed, deadline)
                     if ratio > best_ratio:
                         best_ratio = ratio
                         best_P = P

    # Final validation and normalization
    P = np.clip(best_P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((_N, _N)) / (_N * _N)
    else:
        P = P / s
        
    return P

# EVOLVE-BLOCK-END
