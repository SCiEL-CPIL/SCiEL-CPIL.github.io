import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # Safety margin before 120s (or 150s if evaluator allows)

def _ratio(h):
    """
    Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    Handles scale invariance by normalizing L1=1 internally for evaluation.
    """
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-10:
        return 0.0
    
    h_norm = h / l1
    l2_sq = np.dot(h_norm, h_norm)
    
    # Discrete convolution
    conv = np.convolve(h_norm, h_norm, mode='full')
    max_conv = np.max(conv)
    
    if max_conv < 1e-10:
        return 0.0
        
    return l2_sq / max_conv

def _neg_ratio(h):
    """Objective function for minimization."""
    r = _ratio(h)
    if r <= 0:
        return 1e10
    return -np.log(r)

def _count_sum_collisions(indices):
    """
    Count the number of pairs (i, j) with i <= j such that i+j equals another pair's sum.
    Lower is better. 0 means perfect Sidon set.
    """
    sums = {}
    count = 0
    for i in indices:
        for j in indices:
            if i > j:
                continue
            s = i + j
            if s in sums:
                count += 1
            else:
                sums[s] = 1
    return count

def _optimize_fixed_support(support_indices, start_time, max_iter=100):
    """
    Optimize heights for a fixed set of non-zero indices.
    """
    M = len(support_indices)
    if M == 0:
        return np.zeros(N), 0.0
        
    # Initial guess: equal heights
    h0 = np.zeros(N)
    h0[support_indices] = 1.0 / M
    
    # Define objective that only varies the active indices
    # To use L-BFGS-B efficiently, we can optimize just the active components
    # and map them back.
    
    def objective_active(x_active):
        h = np.zeros(N)
        h[support_indices] = x_active
        return _neg_ratio(h)
    
    x0 = np.full(M, 1.0 / M)
    bounds = [(0, None)] * M
    
    try:
        res = minimize(objective_active, x0, method='L-BFGS-B', 
                       bounds=bounds, 
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-9})
        
        h_opt = np.zeros(N)
        h_opt[support_indices] = np.maximum(res.x, 0.0)
        
        # Re-normalize for consistent output? 
        # The ratio is scale invariant, so normalization doesn't change the score.
        # But returning L1=1 is standard.
        l1 = np.sum(h_opt)
        if l1 > 0:
            h_opt = h_opt / l1
            
        score = _ratio(h_opt)
        return h_opt, score
        
    except Exception:
        return h0, _ratio(h0)

def construct():
    start_time = time.time()
    
    best_h = np.ones(N) / N
    best_score = _ratio(best_h)
    
    # We expect the best score to be around M/2 where M is the Sidon set size.
    # Max M for N=50 is roughly 7-8. So target score ~ 4.0.
    
    # Strategy:
    # 1. Try M from 1 to 10.
    # 2. For each M, generate many random subsets.
    # 3. Rank subsets by collision count (lower is better).
    # 4. Optimize the top K subsets.
    
    rng = np.random.RandomState(42)
    
    # Pre-compute indices
    all_indices = np.arange(N)
    
    max_m_to_try = 10
    trials_per_m = 200
    
    for m in range(1, max_m_to_try + 1):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        candidates = []
        
        # Generate random subsets of size m
        for _ in range(trials_per_m):
            if time.time() - start_time > DEADLINE - 5:
                break
            subset = rng.choice(all_indices, m, replace=False)
            subset.sort()
            colls = _count_sum_collisions(subset)
            candidates.append((colls, subset))
            
        # Sort by collision count (ascending)
        candidates.sort(key=lambda x: x[0])
        
        # Optimize the top few candidates with low collisions
        # If a perfect Sidon set (0 collisions) is found, we can stop optimizing heights
        # as equal heights are optimal, but let's verify with optimizer just in case.
        
        num_to_optimize = min(5, len(candidates))
        
        for i in range(num_to_optimize):
            if time.time() - start_time > DEADLINE - 2:
                break
                
            colls, subset = candidates[i]
            
            # If we have a perfect Sidon set, equal heights give R = M/2.
            # We can calculate this directly without optimization to save time,
            # but optimization might find a slightly better non-equal distribution 
            # if the discrete convolution has specific quirks.
            
            # Quick check: if colls == 0, R should be close to M/2.
            h_equal = np.zeros(N)
            h_equal[subset] = 1.0 / m
            score_equal = _ratio(h_equal)
            
            if score_equal > best_score:
                best_score = score_equal
                best_h = h_equal.copy()
                
            # Run local optimization to see if we can beat the equal-height solution
            h_opt, score_opt = _optimize_fixed_support(subset, start_time, max_iter=50)
            
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
            # If we found a perfect Sidon set and the score is as expected (M/2),
            # we might not need to try other subsets of this size m, 
            # unless we want to see if a different arrangement allows higher M later.
            # But for a fixed m, any Sidon set gives the same theoretical max.
            # So if colls == 0, we can break the inner loop for this m.
            if colls == 0:
                break

    # Final polish: try a global optimization from the best found structure
    # This might help if the discrete support search missed a good configuration
    # or if mixing supports helps.
    
    if time.time() - start_time < DEADLINE - 10:
        # Use the best_h as initial guess for a full vector optimization
        # This is risky as it might collapse, but we keep the best of both.
        try:
            def full_objective(h_flat):
                return _neg_ratio(h_flat)
            
            res = minimize(full_objective, best_h, method='L-BFGS-B', 
                           bounds=[(0, None)] * N,
                           options={'maxiter': 100, 'ftol': 1e-12})
            
            h_final = np.maximum(res.x, 0.0)
            l1 = np.sum(h_final)
            if l1 > 0:
                h_final = h_final / l1
            score_final = _ratio(h_final)
            
            if score_final > best_score:
                best_score = score_final
                best_h = h_final.copy()
        except Exception:
            pass

    # Ensure non-negative and return list
    best_h = np.maximum(best_h, 0.0)
    l1 = np.sum(best_h)
    if l1 > 0:
        best_h = best_h / l1
        
    return list(best_h)
