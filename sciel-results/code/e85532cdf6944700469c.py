import numpy as np
import time
import itertools
from scipy.optimize import minimize
import random

N = 50

def _exact_ratio(h):
    """
    Compute the exact ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    """
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _optimize_weights_on_support(support, deadline):
    """
    Given a fixed set of indices (support), optimize the weights to maximize the ratio.
    We use a parameterization where weights are positive.
    To handle the non-differentiability of max, we use a smooth approximation in the objective
    but evaluate with the exact ratio for final selection.
    
    We use L-BFGS-B on the log-weights to ensure positivity.
    """
    k = len(support)
    if k == 0:
        return None, 0.0
        
    # Initial guess: equal weights
    w_init = np.zeros(k)
    
    def _neg_ratio_smooth(params, beta=50.0):
        # params are log weights
        # Clip to avoid overflow/underflow in exp
        params_clipped = np.clip(params, -20, 20)
        w_local = np.exp(params_clipped)
        # Normalize to sum to 1 (scale invariance)
        s = np.sum(w_local)
        if s < 1e-15:
            return 1e6
        w_local = w_local / s
        
        h_local = np.zeros(N)
        h_local[support] = w_local
        
        l2_sq = np.sum(h_local**2)
        
        # Autoconvolution
        conv = np.convolve(h_local, h_local, mode='full')
        
        # Smooth max
        m = np.max(conv)
        if m < 1e-12:
            return 1e6
        smooth_max = m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta
        
        ratio = l2_sq / (1.0 * smooth_max) # l1 is 1.0
        return -ratio

    try:
        res = minimize(
            _neg_ratio_smooth,
            w_init,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        w_opt_log = res.x
        w_opt = np.exp(np.clip(w_opt_log, -20, 20))
        w_opt = w_opt / np.sum(w_opt)
        
        h_full = np.zeros(N)
        h_full[support] = w_opt
        
        score = _exact_ratio(h_full)
        return h_full, score
        
    except Exception:
        return None, 0.0

def _find_best_supports(deadline, num_to_find=20):
    """
    Heuristically find supports that minimize the peak of the autoconvolution.
    A good support has distinct pairwise sums (i+j).
    This is related to Sidon sets.
    We use a greedy construction with random restarts.
    """
    best_supports = []
    
    # Try a few different random seeds and greedy strategies
    for seed in range(10):
        if time.time() > deadline - 5:
            break
        random.seed(seed)
        np.random.seed(seed)
        
        support = []
        # Greedy addition
        # Start with a random index
        support.append(random.randint(0, N-1))
        
        # Try to add indices that maximize the "spread" of sums
        # Or simply minimize the max collision count of sums
        for _ in range(10): # Target support size ~10-15
            if time.time() > deadline - 5:
                break
            best_idx = None
            best_score = -1
            # Sample candidates
            candidates = random.sample(range(N), min(20, N))
            for cand in candidates:
                if cand in support:
                    continue
                test_support = support + [cand]
                # Evaluate "goodness": minimize number of duplicate sums
                sums = [i+j for i in test_support for j in test_support]
                # We want distinct sums. Count duplicates.
                # Actually, we want to minimize the max frequency of any sum.
                # But for ratio, we want to minimize max_conv.
                # max_conv is sum of h_i h_j where i+j=k.
                # If we assume equal weights, max_conv is proportional to max frequency of sum k.
                freq = {}
                for s in sums:
                    freq[s] = freq.get(s, 0) + 1
                max_freq = max(freq.values()) if freq else 0
                
                # We want low max_freq. 
                # Let's use -max_freq as score (higher is better)
                score = -max_freq
                if score > best_score:
                    best_score = score
                    best_idx = cand
                    
            if best_idx is not None:
                support.append(best_idx)
            else:
                break
                
        # Sort support for consistency
        support.sort()
        if support not in best_supports:
            best_supports.append(support)
            
        if len(best_supports) >= num_to_find:
            break
            
    # Also add some random supports of varying sizes
    for size in [5, 8, 12, 15, 20]:
        if time.time() > deadline - 5:
            break
        for _ in range(5):
            supp = random.sample(range(N), size)
            supp.sort()
            if supp not in best_supports:
                best_supports.append(supp)
                
    return best_supports

def construct():
    start_time = time.time()
    deadline = start_time + 140 # Safety margin for 150s limit
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # 1. Generate candidate supports
    supports = _find_best_supports(deadline, num_to_find=30)
    
    # 2. Optimize weights for each support
    for supp in supports:
        if time.time() > deadline - 10:
            break
        h_candidate, score = _optimize_weights_on_support(supp, deadline)
        if h_candidate is not None and score > best_score:
            best_score = score
            best_h = h_candidate.copy()
            
    # 3. If the sparse approach didn't yield a high score, try a few dense random restarts
    #    with Nelder-Mead on a reduced dimension (e.g., parametric family)
    if best_score < 1.1:
        # Try a Gaussian-like bump with variable width and center
        def _gauss_ratio(params):
            center, width, amp = params
            x = np.linspace(-1, 1, N)
            h = amp * np.exp(-((x - center)**2) / (2 * width**2))
            # Ensure non-negative (exp is always positive)
            # Scale invariance: amp doesn't matter for ratio, but let's keep it for stability
            return -_exact_ratio(h)
        
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            c0 = random.uniform(-0.5, 0.5)
            w0 = random.uniform(0.1, 0.5)
            a0 = 1.0
            try:
                res = minimize(
                    _gauss_ratio,
                    [c0, w0, a0],
                    method='Nelder-Mead',
                    options={'maxiter': 200}
                )
                c_opt, w_opt, a_opt = res.x
                x = np.linspace(-1, 1, N)
                h_gauss = a_opt * np.exp(-((x - c_opt)**2) / (2 * w_opt**2))
                score_g = _exact_ratio(h_gauss)
                if score_g > best_score:
                    best_score = score_g
                    best_h = h_gauss.copy()
            except Exception:
                pass

    # Fallback
    if best_score < 0.1:
        best_h = np.ones(N) / N
        best_score = _exact_ratio(best_h)
        
    # Ensure non-negative
    best_h = np.maximum(best_h, 0)
    
    # Normalize for output stability (ratio is scale invariant, but avoids huge/small numbers)
    if np.sum(best_h) > 0:
        best_h = best_h / np.sum(best_h)
        
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print(f"Score: {_exact_ratio(np.array(h))}")
