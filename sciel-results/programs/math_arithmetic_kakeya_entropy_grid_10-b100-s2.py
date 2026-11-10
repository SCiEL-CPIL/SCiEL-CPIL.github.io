import numpy as np
from scipy.optimize import minimize
import time

# Precompute projection matrices for fast score computation
_N = 10
_YS, _XS = np.indices((_N, _N))

# D: x - y, range -9 to 9 -> 19 bins
_DIFF_IDX = (_XS - _YS + 9).ravel()  # shape (100,)
_DiffMat = np.zeros((19, 100))
for i, idx in enumerate(_DIFF_IDX):
    _DiffMat[idx, i] = 1.0

# H(x): x, range 0 to 9 -> 10 bins
_X_IDX = _XS.ravel()
_XMat = np.zeros((10, 100))
for i, idx in enumerate(_X_IDX):
    _XMat[idx, i] = 1.0

# H(x+y): range 0 to 18 -> 19 bins
_SUM_IDX = (_XS + _YS).ravel()
_SumMat = np.zeros((19, 100))
for i, idx in enumerate(_SUM_IDX):
    _SumMat[idx, i] = 1.0

# H(x+2y): range 0 to 27 -> 28 bins
_SUM2_IDX = (_XS + 2 * _YS).ravel()
_Sum2Mat = np.zeros((28, 100))
for i, idx in enumerate(_SUM2_IDX):
    _Sum2Mat[idx, i] = 1.0

def _shannon_entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 1e-15]
    if len(p) == 0:
        return 0.0
    return -np.sum(p * np.log2(p))

def _compute_score(P):
    """Compute D/M ratio from full 10x10 P matrix."""
    P_flat = P.ravel()
    
    diff_counts = _DiffMat @ P_flat
    D = _shannon_entropy(diff_counts)
    
    x_counts = _XMat @ P_flat
    Hx = _shannon_entropy(x_counts)
    
    sum_counts = _SumMat @ P_flat
    Hs = _shannon_entropy(sum_counts)
    
    sum2_counts = _Sum2Mat @ P_flat
    Hv = _shannon_entropy(sum2_counts)
    
    M = max(Hx, Hs, Hv)
    
    if M <= 1e-9:
        return -np.inf
        
    return D / M

def _score_sparse(support, weights):
    """Compute score for a sparse distribution defined by support and weights."""
    P = np.zeros((_N, _N))
    for i, (x, y) in enumerate(support):
        P[y, x] = weights[i]
    return _compute_score(P)

def _score_sparse_fast(support, weights):
    """Faster: build P flat vector directly."""
    P_flat = np.zeros(100)
    for i, (x, y) in enumerate(support):
        P_flat[y * _N + x] = weights[i]
    
    diff_counts = _DiffMat @ P_flat
    D = _shannon_entropy(diff_counts)
    
    x_counts = _XMat @ P_flat
    Hx = _shannon_entropy(x_counts)
    
    sum_counts = _SumMat @ P_flat
    Hs = _shannon_entropy(sum_counts)
    
    sum2_counts = _Sum2Mat @ P_flat
    Hv = _shannon_entropy(sum2_counts)
    
    M = max(Hx, Hs, Hv)
    
    if M <= 1e-9:
        return -np.inf
        
    return D / M

def _greedy_build_support(seed=42, max_size=30, time_limit=55.0):
    """
    Greedy submodular maximization heuristic to build a support set.
    We want to maximize D/M. We iteratively add the cell that gives the best
    marginal gain in D while penalizing increases in M.
    """
    n = 10
    all_cells = [(x, y) for y in range(n) for x in range(n)]
    
    start_time = time.time()
    
    def optimize_weights(support, w0=None):
        if not support:
            return np.array([])
        k = len(support)
        if w0 is None:
            w0 = np.ones(k) / k
        
        def objective(w):
            w = np.maximum(w, 0)
            s = np.sum(w)
            if s <= 1e-15:
                return 1e10
            w = w / s
            score = _score_sparse_fast(support, w)
            if score == -np.inf:
                return 1e10
            return -score
        
        try:
            res = minimize(objective, w0, method='Nelder-Mead', 
                           options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 2000, 'maxfev': 2000})
            w_opt = np.maximum(res.x, 0)
            s = np.sum(w_opt)
            if s <= 1e-15:
                return w0
            return w_opt / s
        except:
            return w0
    
    # Start with a random support of size 5
    np.random.seed(seed)
    start_cells = np.random.choice(len(all_cells), 5, replace=False)
    support = [all_cells[i] for i in start_cells]
    weights = optimize_weights(support)
    current_score = _score_sparse_fast(support, weights)
    
    no_improve_count = 0
    threshold = 1e-4
    
    while len(support) < max_size:
        if time.time() - start_time > time_limit:
            break
        
        best_gain = 0
        best_cell = None
        best_weights = None
        
        # Try adding each cell not already in support
        support_set = set(support)
        for cell in all_cells:
            if cell in support_set:
                continue
            if time.time() - start_time > time_limit:
                break
            
            new_support = support + [cell]
            # Initialize weights: keep old weights, give tiny weight to new cell
            new_w0 = np.append(weights, 0.01)
            new_weights = optimize_weights(new_support, new_w0)
            new_score = _score_sparse_fast(new_support, new_weights)
            
            gain = new_score - current_score
            if gain > best_gain + threshold:
                best_gain = gain
                best_cell = cell
                best_weights = new_weights
        
        if best_cell is not None:
            support.append(best_cell)
            weights = best_weights
            current_score += best_gain
            no_improve_count = 0
        else:
            break
    
    return support, weights, current_score

def _final_polish(P, max_iter=5000):
    """High-precision polish using SLSQP on the full 100-dim probability simplex."""
    n = 10
    w = P.ravel().copy()
    w = np.maximum(w, 0)
    s = np.sum(w)
    if s <= 1e-15:
        return P
    w = w / s
    
    def objective(w):
        w = np.maximum(w, 0)
        sw = np.sum(w)
        if sw <= 1e-15:
            return 1e10
        w = w / sw
        Pmat = w.reshape(n, n)
        score = _compute_score(Pmat)
        if score == -np.inf:
            return 1e10
        return -score
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * (n * n)
    
    try:
        res = minimize(objective, w, method='SLSQP', bounds=bounds,
                       constraints=constraints, options={'maxiter': max_iter, 'ftol': 1e-12})
        w_opt = np.maximum(res.x, 0)
        s = np.sum(w_opt)
        if s > 1e-15:
            w_opt = w_opt / s
            return w_opt.reshape(n, n)
    except:
        pass
    
    return P

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 615  # 15s safety margin before 630s kill
    
    n = 10
    best_score = 0.0
    best_P = np.ones((n, n)) / (n * n)
    
    # Initialize with uniform
    score_uniform = _compute_score(best_P)
    best_score = score_uniform
    
    # Try several random restarts of the greedy heuristic
    num_restarts = 12
    
    for r in range(num_restarts):
        if time.time() > deadline - 60:  # reserve 60s for polish
            break
        
        # Different seeds for diversity
        support, weights, score = _greedy_build_support(seed=42 + r * 7, max_size=30, time_limit=40.0)
        
        if score > best_score:
            best_score = score
            P = np.zeros((n, n))
            for i, (x, y) in enumerate(support):
                P[y, x] = weights[i]
            best_P = P
    
    # Final polish on the best found distribution
    if time.time() < deadline - 5:
        best_P = _final_polish(best_P, max_iter=3000)
        polished_score = _compute_score(best_P)
        if polished_score > best_score:
            best_score = polished_score
    
    # Final check
    if best_P.sum() <= 1e-9 or np.any(~np.isfinite(best_P)):
        return np.ones((n, n)) / (n * n)
    
    # Clip and normalize
    best_P = np.maximum(best_P, 0)
    s = np.sum(best_P)
    if s > 1e-9:
        best_P = best_P / s
    
    return best_P
