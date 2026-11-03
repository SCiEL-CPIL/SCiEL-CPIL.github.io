import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    """
    Computes the score: L2^2 / (L1 * Sup(AutoConv))
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    
    if autoconv_max == 0:
        return 0.0
        
    return l2_sq / (l1 * autoconv_max)

def _soft_max_conv(h, tau=1e-3):
    """
    Smooth approximation of max(h*h)
    """
    h_conv = np.convolve(h, h, mode='full')
    max_val = np.max(h_conv)
    if max_val == 0:
        return 0.0
    # log-sum-exp trick for stability
    log_sum_exp = max_val + tau * np.log(np.sum(np.exp((h_conv - max_val) / tau)))
    return log_sum_exp

def _objective_weights(weights, indices, tau=1e-3):
    """
    Objective for SLSQP on a fixed support.
    Minimizes -Score.
    """
    h = np.zeros(N)
    h[indices] = weights
    l1 = np.sum(h)
    if l1 < 1e-10:
        return 1e10 # Penalty for zero sum
    
    l2_sq = np.sum(h**2)
    smc = _soft_max_conv(h, tau)
    
    if smc < 1e-10:
        return 1e10
        
    score = l2_sq / (l1 * DELTA * smc)
    return -score

def _optimize_weights_for_support(indices, h_init=None, max_iter=200):
    """
    Optimizes weights for a given set of indices.
    Returns optimized weights and score.
    """
    k = len(indices)
    if k == 0:
        return None, 0.0
    
    # Initialize weights
    if h_init is not None:
        w0 = h_init[indices].copy()
    else:
        w0 = np.ones(k) / k
    
    # Ensure non-negative
    w0 = np.clip(w0, 1e-6, None)
    
    # Normalize for initial guess stability
    w0 = w0 / np.sum(w0)
    
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    ]
    bounds = [(0.0, None)] * k
    
    # Use a small tau for better gradient signal, but anneal if possible
    # For simplicity, fixed small tau in local search
    tau = 1e-3 
    
    try:
        res = minimize(
            _objective_weights, 
            w0, 
            args=(indices, tau),
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints, 
            options={'maxiter': max_iter, 'ftol': 1e-10}
        )
        
        if res.success:
            w_opt = np.clip(res.x, 0, None)
            if np.sum(w_opt) == 0:
                return None, 0.0
            w_opt = w_opt / np.sum(w_opt)
            
            # Construct full h to get raw score
            h_full = np.zeros(N)
            h_full[indices] = w_opt
            score = _score_raw(h_full)
            return w_opt, score
    except Exception:
        pass
        
    # Fallback: return initial
    h_full = np.zeros(N)
    h_full[indices] = w0
    score = _score_raw(h_full)
    return w0, score

def _greedy_sidon(n, k):
    """
    Greedy construction of a Sidon-like set (distinct differences).
    """
    S = []
    used_diffs = set()
    for i in range(n):
        if len(S) >= k:
            break
        # Check if i can be added
        valid = True
        new_diffs = set()
        for s in S:
            d = abs(i - s)
            if d in used_diffs:
                valid = False
                break
            new_diffs.add(d)
        if valid:
            S.append(i)
            used_diffs.update(new_diffs)
    return S

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # 1. Generate Initial Supports
    supports = []
    
    # Uniform
    supports.append(np.arange(N))
    
    # Greedy Sidon sets of various sizes
    for k in [5, 8, 10, 12, 15, 20]:
        if time.time() > deadline - 10: break
        S = _greedy_sidon(N, k)
        if len(S) >= 2:
            supports.append(np.array(S))
            
    # Random sparse sets
    np.random.seed(42)
    for _ in range(20):
        if time.time() > deadline - 10: break
        k = np.random.randint(5, 15)
        S = np.random.choice(N, k, replace=False)
        supports.append(S)
        
    # 2. Optimize each support
    # We will iterate through supports, optimize, and keep the best.
    # We also perform a local search on the support of the best solution.
    
    current_best_support = None
    current_best_weights = None
    
    for S in supports:
        if time.time() > deadline - 5:
            break
            
        # Optimize weights for this support
        w_opt, score = _optimize_weights_for_support(S, max_iter=100)
        
        if w_opt is not None and score > best_score:
            best_score = score
            # Reconstruct h for reporting
            h_temp = np.zeros(N)
            h_temp[S] = w_opt
            best_h = h_temp
            current_best_support = S.copy()
            current_best_weights = w_opt.copy()
            
    # 3. Local Search on the Best Support
    # Try to add/remove/swap indices to improve the score further.
    if current_best_support is not None and time.time() < deadline - 10:
        # Start from the best found
        S_curr = current_best_support.copy()
        w_curr = current_best_weights.copy()
        score_curr = best_score
        
        # List of all indices
        all_indices = np.arange(N)
        
        # Iterate a few times
        for _ in range(50):
            if time.time() > deadline - 5:
                break
                
            improved = False
            
            # Try adding a random index not in S_curr
            if len(S_curr) < N - 1:
                candidates_to_add = np.setdiff1d(all_indices, S_curr)
                if len(candidates_to_add) > 0:
                    # Pick a few random candidates
                    num_trials = min(10, len(candidates_to_add))
                    trial_indices = np.random.choice(candidates_to_add, num_trials, replace=False)
                    
                    for idx in trial_indices:
                        new_S = np.append(S_curr, idx)
                        # Initialize new weight as small
                        new_w = np.append(w_curr, 0.01)
                        # Re-optimize
                        w_opt, score_new = _optimize_weights_for_support(new_S, h_init=np.zeros(N).copy(), max_iter=50)
                        # Wait, _optimize_weights_for_support takes indices and optional h_init (full vector)
                        # We need to pass the current weights as initialization for the new vector
                        h_init_full = np.zeros(N)
                        h_init_full[S_curr] = w_curr
                        h_init_full[idx] = 0.01
                        
                        w_opt, score_new = _optimize_weights_for_support(new_S, h_init=h_init_full, max_iter=50)
                        
                        if w_opt is not None and score_new > score_curr:
                            S_curr = new_S
                            w_curr = w_opt
                            score_curr = score_new
                            best_score = score_curr
                            h_temp = np.zeros(N)
                            h_temp[S_curr] = w_curr
                            best_h = h_temp
                            improved = True
                            break # Restart loop with new best
                    
            # Try removing an index from S_curr
            if len(S_curr) > 2:
                # Pick a few random indices to try removing
                num_trials = min(10, len(S_curr))
                trial_indices = np.random.choice(len(S_curr), num_trials, replace=False)
                
                for pos in trial_indices:
                    idx_to_remove = S_curr[pos]
                    new_S = np.delete(S_curr, pos)
                    new_w = np.delete(w_curr, pos)
                    
                    h_init_full = np.zeros(N)
                    h_init_full[new_S] = new_w
                    
                    w_opt, score_new = _optimize_weights_for_support(new_S, h_init=h_init_full, max_iter=50)
                    
                    if w_opt is not None and score_new > score_curr:
                        S_curr = new_S
                        w_curr = w_opt
                        score_curr = score_new
                        best_score = score_curr
                        h_temp = np.zeros(N)
                        h_temp[S_curr] = w_curr
                        best_h = h_temp
                        improved = True
                        break # Restart loop with new best
                        
            if not improved:
                # No improvement in this round, maybe break early or continue with different randoms
                if time.time() > deadline - 20:
                    break

    # 4. Final Polish with SLSQP on the full vector (if support is dense or just to clean up)
    # If the support is small, SLSQP on full vector might be slow or get stuck.
    # But if we have a good h, we can try a final global SLSQP with the best h as start.
    if time.time() < deadline - 5:
        try:
            # Use the best_h as start
            # We need to handle the non-smooth max.
            # We'll use the soft-max objective on the full vector.
            constraints = [
                {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}
            ]
            bounds = [(0.0, 1.0)] * N
            
            # Define objective for full vector
            def full_obj(h):
                l1 = np.sum(h)
                if l1 < 1e-10: return 1e10
                l2_sq = np.sum(h**2)
                smc = _soft_max_conv(h, 1e-4)
                if smc < 1e-10: return 1e10
                score = l2_sq / (l1 * DELTA * smc)
                return -score
            
            res = minimize(full_obj, best_h, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 100, 'ftol': 1e-12})
            
            if res.success:
                h_candidate = np.clip(res.x, 0, None)
                s_norm = np.sum(h_candidate)
                if s_norm > 0:
                    h_candidate /= s_norm
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    return list(best_h)

# Run once to ensure no syntax errors
if __name__ == "__main__":
    h = construct()
    print("Score:", _score_raw(h))
