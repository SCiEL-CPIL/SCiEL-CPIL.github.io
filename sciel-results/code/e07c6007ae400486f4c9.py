import numpy as np
from scipy.optimize import minimize
import time

def _shannon_entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 1e-15]
    if len(p) == 0:
        return 0.0
    return -np.sum(p * np.log2(p))

def _compute_score(P):
    n = 10
    ys, xs = np.indices((n, n))
    
    # D = H(x - y)
    diffs = xs - ys  # range -9 to 9
    diff_counts = np.zeros(19)
    for d in range(-9, 10):
        mask = (diffs == d)
        diff_counts[d + 9] = np.sum(P[mask])
    D = _shannon_entropy(diff_counts)
    
    # M = max(H(x), H(x+y), H(x+2y))
    # H(x)
    x_counts = np.zeros(n)
    for x in range(n):
        x_counts[x] = np.sum(P[:, x])
    Hx = _shannon_entropy(x_counts)
    
    # H(x+y)
    sum_counts = np.zeros(19)
    for s in range(19):
        mask = (xs + ys == s)
        sum_counts[s] = np.sum(P[mask])
    Hs = _shannon_entropy(sum_counts)
    
    # H(x+2y)
    sum2_counts = np.zeros(28)
    for v in range(28):
        mask = (xs + 2 * ys == v)
        sum2_counts[v] = np.sum(P[mask])
    Hv = _shannon_entropy(sum2_counts)
    
    M = max(Hx, Hs, Hv)
    
    if M <= 1e-9:
        return -np.inf
        
    return D / M

def _score_sparse(support, weights):
    """Compute score for a sparse distribution defined by support and weights."""
    n = 10
    P = np.zeros((n, n))
    for i, (x, y) in enumerate(support):
        P[y, x] = weights[i]
    return _compute_score(P)

def _greedy_build_support():
    """
    Greedy submodular maximization heuristic to build a support set.
    We want to maximize D/M. We iteratively add the cell that gives the best
    marginal gain in D while penalizing increases in M.
    """
    n = 10
    all_cells = [(x, y) for y in range(n) for x in range(n)]
    support = []
    weights = []
    
    # We'll build a support of size roughly 19-30.
    # Start with an empty distribution.
    # To avoid division by zero, we start with a tiny uniform distribution or just add the first cell.
    
    # Let's try to maximize D/M directly.
    # Since M is in the denominator, we want to keep M low.
    # D is in the numerator, we want to keep D high.
    
    # A good strategy might be to pick cells that spread out x-y values (high D)
    # but concentrate x, x+y, x+2y values (low M).
    
    # Let's use a simple greedy approach:
    # Start with a random seed cell.
    # Then iteratively pick the cell that maximizes the score of the current support + new cell,
    # with weights optimized for the current support + new cell.
    
    # This is expensive, so let's use a faster proxy or limit the search.
    
    # Alternative: Start with a known good structure.
    # The diagonal x=y gives x-y=0, so D=0. Bad.
    # Anti-diagonal x+y=9 gives x-y = x-(9-x) = 2x-9.
    # x=0->-9, x=1->-7, ..., x=9->9.
    # Diffs: -9, -7, -5, -3, -1, 1, 3, 5, 7, 9.
    # These are 10 distinct values.
    # If we put equal weight on these, D = log2(10) ~ 3.32.
    # Let's check M for anti-diagonal.
    # Cells: (0,9), (1,8), ..., (9,0).
    # x values: 0..9 (all distinct). H(x) = log2(10).
    # x+y = 9 (all same). H(x+y) = 0.
    # x+2y = x + 2(9-x) = 18 - x. Values 18..9 (distinct). H(x+2y) = log2(10).
    # So M = log2(10).
    # Ratio D/M = 1.
    
    # Can we do better?
    # If we could make D > log2(10), but max distinct diffs is 19.
    # If we use 19 cells with distinct diffs, D can be up to log2(19) ~ 4.25.
    # But M might increase.
    
    # Let's try a support that has distinct diffs but tries to keep x, x+y, x+2y concentrated.
    
    # Greedy construction:
    # 1. Initialize support with a few random cells or a heuristic start.
    # 2. At each step, try adding every possible new cell.
    # 3. For each candidate new cell, optimize weights on the new support (size k+1) using a fast method.
    # 4. Pick the cell that maximizes the score.
    # 5. Repeat until no improvement or max size reached.
    
    # To speed up, we can limit the number of candidates or use a faster weight optimization.
    
    # Let's try a different structural idea:
    # Maximize D by ensuring many distinct x-y values.
    # Minimize M by ensuring x, x+y, x+2y values are NOT all distinct.
    
    # Consider a set of cells where x is constant.
    # If x=5, then x-y = 5-y. As y varies 0..9, diffs are 5..-4 (10 distinct).
    # D = log2(10).
    # H(x) = 0.
    # x+y = 5+y. Values 5..14 (10 distinct). H(x+y) = log2(10).
    # x+2y = 5+2y. Values 5,7,9,11,13,15,17,19,21,23 (10 distinct). H(x+2y) = log2(10).
    # M = log2(10). Ratio = 1.
    
    # What if we mix?
    
    # Let's implement the greedy support building with weight optimization.
    
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
            P = np.zeros((n, n))
            for i, (x, y) in enumerate(support):
                P[y, x] = w[i]
            return -_compute_score(P)
        
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
    np.random.seed(42)
    start_cells = np.random.choice(len(all_cells), 5, replace=False)
    support = [all_cells[i] for i in start_cells]
    weights = optimize_weights(support)
    current_score = _score_sparse(support, weights)
    
    max_size = 25
    no_improve_count = 0
    threshold = 1e-4
    
    while len(support) < max_size:
        best_gain = 0
        best_cell = None
        best_weights = None
        
        # Try adding each cell not already in support
        support_set = set(support)
        for cell in all_cells:
            if cell in support_set:
                continue
            
            new_support = support + [cell]
            # Initialize weights: keep old weights, give tiny weight to new cell
            new_w0 = np.append(weights, 0.01)
            new_weights = optimize_weights(new_support, new_w0)
            new_score = _score_sparse(new_support, new_weights)
            
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

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # 30s margin
    
    n = 10
    best_score = 0.0
    best_P = np.ones((n, n)) / (n * n)
    
    # Initialize with uniform
    score_uniform = _compute_score(best_P)
    best_score = score_uniform
    
    # Try several random restarts of the greedy heuristic
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different seeds for diversity
        np.random.seed(42 + r)
        
        support, weights, score = _greedy_build_support()
        
        if score > best_score:
            best_score = score
            P = np.zeros((n, n))
            for i, (x, y) in enumerate(support):
                P[y, x] = weights[i]
            best_P = P
        
        # Refine the best found so far with a final optimization
        if score > best_score - 0.01:
            # Try a few more optimization steps on the current support
            pass
            
    # Final check
    if best_P.sum() <= 1e-9 or np.any(~np.isfinite(best_P)):
        return np.ones((n, n)) / (n * n)
    
    # Clip and normalize
    best_P = np.maximum(best_P, 0)
    s = np.sum(best_P)
    if s > 1e-9:
        best_P = best_P / s
    
    return best_P
