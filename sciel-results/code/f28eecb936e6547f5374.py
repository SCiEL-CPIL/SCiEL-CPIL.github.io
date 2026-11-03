import time
import numpy as np
from scipy.optimize import minimize

N = 50

def _find_sidon_set(n, max_time=5.0):
    """
    Finds a large Sidon set in {0, ..., n-1} using a greedy approach with random restarts.
    A Sidon set has all pairwise sums i+j (i<=j) distinct.
    """
    best_set = []
    start_time = time.time()
    
    # Try multiple random orderings to escape local optima of greedy
    for _ in range(200):
        if time.time() - start_time > max_time:
            break
        order = np.random.permutation(n)
        S = []
        used_sums = set()
        
        for x in order:
            valid = True
            new_sums = set()
            
            # Check sums with existing elements
            for y in S:
                s = x + y
                if s in used_sums:
                    valid = False
                    break
                new_sums.add(s)
            
            # Check double
            if valid:
                s = 2 * x
                if s in used_sums:
                    valid = False
                else:
                    new_sums.add(s)
            
            if valid:
                S.append(x)
                used_sums.update(new_sums)
                
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) > 7: # Heuristic max for N=50
                break

    return sorted(best_set)

def _evaluate(h):
    """Calculate the raw score."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv == 0:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # 1. Find a large Sidon set
    sidon_set = _find_sidon_set(N, max_time=10.0)
    
    # 2. Construct the optimal weight distribution for this Sidon set
    # Uniform weights are optimal for Sidon sets to maximize the ratio.
    # J = sum(w_i^2) / max(w_i w_j). With uniform w, J = K.
    k = len(sidon_set)
    if k == 0:
        # Fallback: single spike
        h = np.zeros(N)
        h[0] = 1.0
        return h.tolist()
        
    h = np.zeros(N)
    for idx in sidon_set:
        h[idx] = 1.0
        
    # Normalize L1 to 1 to be safe, though the ratio is scale invariant 
    # if the evaluator doesn't scale, but if it does, L1=1 is standard.
    # Actually, the ratio J(h) is NOT scale invariant (scales as 1/c).
    # However, the problem likely implies a normalized context or the "seed" 
    # behavior suggests we should return a vector that yields the max ratio 
    # under the evaluator's specific normalization.
    # Given the seed normalized to L1=1, I will do the same.
    h = h / np.sum(h)
    
    # 3. Optional: Local optimization of weights
    # Although uniform is optimal for Sidon, maybe a non-Sidon set with 
    # optimized weights is better? Unlikely, but we have time.
    # We can try to optimize weights on the Sidon set support.
    
    def neg_score(weights):
        h = np.zeros(N)
        h[sidon_set] = weights
        h = np.abs(h)
        l1 = np.sum(h)
        if l1 == 0: return 1e10
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        sup_conv = np.max(conv)
        if sup_conv == 0: return 1e10
        return -(l2_sq / (l1 * sup_conv))
        
    # Start with uniform weights
    x0 = np.ones(k) / k
    
    try:
        if time.time() < deadline - 20:
            # Optimize only the weights, keeping support fixed
            res = minimize(neg_score, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            h_opt = np.zeros(N)
            h_opt[sidon_set] = res.x
            h_opt = np.abs(h_opt)
            # Normalize
            h_opt = h_opt / np.sum(h_opt)
            
            s_opt = _evaluate(h_opt)
            s_uniform = _evaluate(h)
            
            if s_opt > s_uniform:
                h = h_opt
    except Exception:
        pass

    return h.tolist()
