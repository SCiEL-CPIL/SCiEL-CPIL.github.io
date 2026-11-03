import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _find_sidon_greedy(N, seed=None):
    """Find a large Sidon set (B_2 set) in {0, ..., N-1} using greedy with random restart."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    S = [0]
    # Randomize candidate order
    candidates = list(range(1, N))
    rng.shuffle(candidates)
    
    # Precompute existing sums set for efficiency
    existing_sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            existing_sums.add(S[i] + S[j])
    
    for x in candidates:
        # Check if adding x keeps the set Sidon
        # New sums: x + s for s in S, and x + x
        # We need all new sums to be distinct from existing_sums and from each other
        new_sums = [x + s for s in S] + [2 * x]
        
        # Check for collisions with existing sums
        if any(ns in existing_sums for ns in new_sums):
            continue
        
        # New sums are distinct from each other:
        # x + s1 == x + s2 => s1 == s2 (impossible)
        # x + s1 == 2x => s1 == x (impossible since x not in S)
        
        # Accept x
        S.append(x)
        for ns in new_sums:
            existing_sums.add(ns)
            
    return S

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # The key insight: The ratio R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # is maximized when the support of h forms a Sidon set (B_2 set).
    # For a Sidon set S, all pairwise sums i+j (i<=j) are distinct.
    # This means the autoconvolution (h*h)_k is non-zero at distinct positions for each pair.
    # The maximum of (h*h)_k will be the maximum over pairs of h_i * h_j.
    # If we choose equal weights h_i = c on a Sidon set of size k, then:
    #   ||h||_1 = k*c
    #   ||h||_2^2 = k*c^2
    #   (h*h)_k = c^2 for each of the ~k(k+1)/2 distinct positions
    #   ||h*h||_inf = c^2
    #   R = (k*c^2) / (k*c * c^2) = 1/c
    # Since c = 1/k (if we normalize L1=1), R = k.
    # So the score is approximately the size of the Sidon set!
    # The maximum Sidon set size in {0, ..., 49} is around sqrt(50) ~ 7.
    # This explains why the score can be > 1.
    
    best_h = None
    best_score = 0.0
    
    # Step 1: Find large Sidon sets via random restarts
    best_sidon = []
    for i in range(500):
        if time.time() > deadline:
            break
        S = _find_sidon_greedy(N, seed=i)
        if len(S) > len(best_sidon):
            best_sidon = S
            if len(best_sidon) >= 7:  # sqrt(50) ~ 7.07
                break
    
    # Step 2: Construct h from the best Sidon set with optimized weights
    if best_sidon:
        h_sidon = np.zeros(N)
        h_sidon[best_sidon] = 1.0  # Start with equal weights
        
        # Optimize weights using SLSQP
        # Objective: maximize L2^2 / (L1 * MaxConv)
        # This is scale invariant, so we fix L1 = 1.
        
        def neg_obj(h):
            l1 = np.sum(h)
            if l1 == 0:
                return 0
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv == 0:
                return 0
            return -(l2_sq / (l1 * max_conv))
        
        # Only optimize weights on the Sidon support
        support = np.array(best_sidon)
        idx_in_support = np.where(np.isin(np.arange(N), support))[0]
        
        def neg_obj_reduced(w):
            h = np.zeros(N)
            h[support] = w
            return neg_obj(h)
        
        # Initial weights: equal
        w0 = np.ones(len(support)) / len(support)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * len(support)
        
        if time.time() < deadline - 20:
            res = minimize(neg_obj_reduced, w0, method='SLSQP', bounds=bounds, 
                          constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
            if res.success:
                h_opt = np.zeros(N)
                h_opt[support] = res.x
                s_opt = _score_raw(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt
    
    # Step 3: Also try without strict Sidon constraint - use optimizer from Sidon init
    if best_h is not None and time.time() < deadline - 10:
        h_start = best_h.copy()
        l1 = np.sum(h_start)
        if l1 > 0:
            h_start /= l1
        
        def neg_obj_full(h):
            return neg_obj(h)
        
        constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
        bounds = [(0.0, 1.0)] * N
        
        res = minimize(neg_obj_full, h_start, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-14})
        if res.success:
            h_candidate = res.x
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate
    
    # Fallback if no valid solution found
    if best_h is None or best_score <= 0:
        best_h = np.ones(N) / N
        best_score = _score_raw(best_h)
    
    # Ensure non-negative
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)
