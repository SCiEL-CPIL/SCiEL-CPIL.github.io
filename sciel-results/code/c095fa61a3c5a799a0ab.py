import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 50

def _exact_ratio(h):
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _optimize_weights(fixed_indices, h_full, deadline):
    """
    Optimize weights on a fixed set of indices.
    Returns the best h_full found.
    """
    k = len(fixed_indices)
    if k == 0:
        return h_full
    
    # Initial weights: equal
    w = np.ones(k)
    
    def _neg_ratio_smooth(params, beta=50.0):
        # params are log weights
        w_local = np.exp(params)
        w_local = w_local / np.sum(w_local)
        
        h_local = np.zeros(N)
        h_local[fixed_indices] = w_local
        
        l2_sq = np.sum(h_local**2)
        conv = np.convolve(h_local, h_local, mode='full')
        
        # Smooth max
        m = np.max(conv)
        if m < 0: m = 0
        smooth_max = m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta
        
        if smooth_max < 1e-15:
            return 0.0
        return -l2_sq / smooth_max

    best_score = _exact_ratio(h_full)
    best_w = w.copy()
    
    # Try a few betas for smooth max
    for beta in [20.0, 50.0, 100.0]:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                _neg_ratio_smooth,
                np.log(w + 1e-10),
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            w_res = np.exp(res.x)
            w_res = w_res / np.sum(w_res)
            
            h_res = np.zeros(N)
            h_res[fixed_indices] = w_res
            
            score = _exact_ratio(h_res)
            if score > best_score:
                best_score = score
                best_w = w_res.copy()
        except Exception:
            pass
            
    h_best = np.zeros(N)
    h_best[fixed_indices] = best_w
    return h_best

def _greedy_sidon_search(deadline):
    """
    Greedy search for a support set with low autocorrelation peak.
    We start with a random small set and greedily add elements that minimize the peak autocorrelation.
    """
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Try multiple random starts
    np.random.seed(42)
    
    max_k = 15 # Max number of non-zeros to try
    num_starts = 10
    
    for start_id in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Random initial set of size 3-5
        init_size = 3 + (start_id % 3)
        current_indices = np.random.choice(N, size=init_size, replace=False)
        current_indices.sort()
        
        # Optimize weights on this initial set to get a baseline
        h_curr = np.zeros(N)
        h_curr[current_indices] = 1.0
        h_curr = _optimize_weights(list(current_indices), h_curr, deadline)
        curr_score = _exact_ratio(h_curr)
        
        # Greedy expansion
        for _ in range(max_k - len(current_indices)):
            if time.time() > deadline - 10:
                break
            
            best_new_index = -1
            best_new_score = curr_score
            best_new_h = h_curr.copy()
            
            # Try adding each possible new index
            candidates = [i for i in range(N) if i not in current_indices]
            # To save time, we can sample candidates if N is large, but N=50 is small
            # However, checking all 50 * 10 starts * 15 steps * optimization might be slow.
            # Optimization is expensive. Let's only do weight optimization if the structural score improves.
            
            # First, evaluate structural impact with equal weights quickly
            # Actually, let's just pick the top few candidates based on a quick metric
            # Metric: minimize max autocorrelation of the set (with unit weights)
            
            # Quick metric: sum of squares of autocorrelation? No, we want to minimize peak.
            # Let's just test adding each candidate with equal weights first.
            
            base_h = np.zeros(N)
            base_h[current_indices] = 1.0
            base_conv = np.convolve(base_h, base_h, mode='full')
            base_max = np.max(base_conv)
            
            # We want to add an index that doesn't increase the peak too much, or ideally keeps it low.
            # Actually, for Sidon sets, the peak of the autocorrelation of the indicator function is 1 (at lag 0) if we consider normalized?
            # No, conv of indicator is count of pairs.
            # Let's stick to: find index j such that max(conv(h + e_j)) is minimized.
            
            best_candidate_score = float('inf')
            best_candidate_idx = -1
            
            for j in candidates:
                h_test = base_h.copy()
                h_test[j] = 1.0
                conv_test = np.convolve(h_test, h_test, mode='full')
                max_test = np.max(conv_test)
                # We prefer lower max autocorrelation
                if max_test < best_candidate_score:
                    best_candidate_score = max_test
                    best_candidate_idx = j
            
            if best_candidate_idx == -1:
                break
                
            # Add the best candidate
            new_indices = list(current_indices) + [best_candidate_idx]
            new_indices.sort()
            
            # Optimize weights on the new set
            h_test = np.zeros(N)
            h_test[new_indices] = 1.0
            h_optimized = _optimize_weights(new_indices, h_test, deadline)
            new_score = _exact_ratio(h_optimized)
            
            if new_score > best_new_score:
                curr_score = new_score
                current_indices = new_indices
                h_curr = h_optimized
                
            if curr_score > best_score:
                best_score = curr_score
                best_h = h_curr.copy()
                
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    best_h, best_score = _greedy_sidon_search(deadline)
    
    # If greedy search failed or score is bad, fallback to a simple dense initialization
    if best_score < 0.5:
        h_fallback = np.ones(N) / N
        score_fallback = _exact_ratio(h_fallback)
        if score_fallback > best_score:
            best_h = h_fallback
            best_score = score_fallback
            
    # Final polish: try to optimize the full vector from the best sparse structure found
    # This allows some small non-zero values at other positions if beneficial
    if time.time() < deadline - 5:
        # Use the best_h as init for a full optimization
        # But full optimization is what we are trying to avoid/replace?
        # Actually, the directive says "different optimization or search method".
        # The current program does full optimization.
        # My strategy is: Combinatorial Support Search + Local Weight Optimization.
        # I will NOT do a full 50-dim optimization at the end, to stay true to the strategy.
        # However, a final local step on the full vector initialized by the sparse best might help.
        # Let's do a very short L-BFGS-B on the full vector to "smooth" out any discretization errors?
        # No, that might collapse the sparsity. The sparsity is the key structural feature.
        
        # Instead, let's do a second pass of greedy search with a different seed or strategy if time permits.
        # Or just return the best sparse solution.
        
        # Let's try one more thing: random restarts of the greedy process with different random seeds
        # to see if we find a better support.
        for seed in [1, 2, 3]:
            if time.time() > deadline - 5:
                break
            np.random.seed(seed)
            h_try, score_try = _greedy_sidon_search(deadline)
            if score_try > best_score:
                best_score = score_try
                best_h = h_try
                
    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
