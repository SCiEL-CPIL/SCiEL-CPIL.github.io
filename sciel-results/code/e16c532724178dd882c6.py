import numpy as np
import time
import sys

def evaluate_measure(s, w):
    """
    Fast evaluation of P[X1 + X2 + X3 < 2*X4].
    P = sum_{k,l} p_k * p_l * sum_{i,j: s_i+s_j < 2*s_l - s_k} p_i * p_j
    """
    n = len(s)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 0.0
    p = w / w_sum
    p = np.maximum(p, 0.0)
    s = np.maximum(s, 0.0)
    
    # Re-normalize after clipping
    p_sum = np.sum(p)
    if p_sum <= 1e-9:
        return 0.0
    p = p / p_sum
    
    # Build pair sums and probs
    ss = s[:, None] + s[None, :]
    pp = p[:, None] * p[None, :]
    
    ss_flat = ss.flatten()
    pp_flat = pp.flatten()
    
    # Sort by sum
    sort_idx = np.argsort(ss_flat)
    ss_sorted = ss_flat[sort_idx]
    pp_sorted = pp_flat[sort_idx]
    
    # Cumulative sum of probs
    C = np.cumsum(pp_sorted)
    
    # Vectorized thresholds for all (k, l) pairs
    thresholds = 2.0 * s[None, :] - s[:, None]
    idx = np.searchsorted(ss_sorted, thresholds, side='left')
    # Get cumulative probabilities up to idx-1
    # Handle idx=0 case
    prob_inner = np.zeros_like(idx, dtype=np.float64)
    mask = idx > 0
    prob_inner[mask] = C[idx[mask] - 1]
    
    P = np.sum(p[:, None] * p[None, :] * prob_inner)
            
    return P

def coordinate_ascent_weights(s, w_init, time_limit, seed=42):
    """
    Deterministic coordinate ascent for weights on a fixed support.
    Uses a multi-start approach with structured initializations.
    """
    n = len(s)
    if n == 0:
        return np.array([1.0]), 0.0
    
    s = np.maximum(s, 0.0)
    
    # Precompute sorted pair sums for fast evaluation
    ss = s[:, None] + s[None, :]
    ss_flat = ss.flatten()
    sort_idx = np.argsort(ss_flat)
    ss_sorted = ss_flat[sort_idx]
    
    # Precompute thresholds
    thresholds = 2.0 * s[None, :] - s[:, None]
    idx = np.searchsorted(ss_sorted, thresholds, side='left')
    
    def fast_eval(w):
        p = w / np.sum(w)
        p = np.maximum(p, 0.0)
        p_sum = np.sum(p)
        if p_sum <= 1e-9: return 0.0
        p = p / p_sum
        
        pp_flat = (p[:, None] * p[None, :]).flatten()
        # We need to sum pp_flat[sort_idx] for indices < idx
        # This is equivalent to summing pp_sorted up to idx-1
        # But pp_flat is permuted by sort_idx.
        # Actually, the previous evaluate_measure does:
        # C = cumsum(pp_sorted)
        # prob_inner[k,l] = C[idx[k,l]-1]
        # P = sum(p_k p_l prob_inner[k,l])
        
        # To speed up coordinate ascent, we can update incrementally?
        # No, let's just use the vectorized version but avoid re-sorting.
        # The sort order of ss is fixed.
        # pp_sorted = pp_flat[sort_idx]
        # C = cumsum(pp_sorted)
        
        pp_sorted = pp_flat[sort_idx]
        C = np.cumsum(pp_sorted)
        
        prob_inner = np.zeros_like(idx, dtype=np.float64)
        mask = idx > 0
        prob_inner[mask] = C[idx[mask] - 1]
        
        return np.sum(p[:, None] * p[None, :] * prob_inner)

    best_w = w_init.copy()
    best_score = fast_eval(best_w)
    
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    # Multi-start with different initializations
    inits = [
        w_init,
        np.ones(n) / n,
        np.exp(-np.linspace(0, 3, n)), # Decaying
        np.exp(np.linspace(0, 3, n)),  # Growing
        np.array([0.1, 0.1, 0.1, 0.7]) if n == 4 else np.ones(n)/n
    ]
    
    for w0 in inits:
        if time.time() - start_time > time_limit:
            break
            
        w = w0.copy()
        current_score = fast_eval(w)
        
        if current_score > best_score:
            best_score = current_score
            best_w = w.copy()
            
        # Coordinate Ascent
        improved = True
        step_size = 0.1
        max_iter = 5000
        
        for it in range(max_iter):
            if time.time() - start_time > time_limit:
                return best_w, best_score
            if not improved:
                break
                
            improved = False
            
            # Try moving each coordinate
            for i in range(n):
                # Try small increase and decrease
                for delta in [step_size, -step_size]:
                    if w[i] + delta < 0:
                        continue
                        
                    w_trial = w.copy()
                    w_trial[i] += delta
                    # Renormalize to keep sum constant? 
                    # The metric uses w_i / sum(w). So scaling doesn't matter.
                    # But moving mass from i to j is better for exploration.
                    # Let's do pairwise moves for better geometry.
                    pass
                
                # Pairwise moves
                for j in range(n):
                    if i == j: continue
                    for delta in [step_size, -step_size]:
                        if w[i] + delta < 0 or w[j] - delta < 0:
                            continue
                            
                        w_trial = w.copy()
                        w_trial[i] += delta
                        w_trial[j] -= delta
                        
                        score_trial = fast_eval(w_trial)
                        
                        if score_trial > current_score + 1e-12:
                            w = w_trial
                            current_score = score_trial
                            if current_score > best_score:
                                best_score = current_score
                                best_w = w.copy()
                            improved = True
                            step_size *= 0.95
                            break
                    if improved:
                        break
                if improved:
                    break
                    
            if not improved:
                step_size *= 0.7
                if step_size < 1e-5:
                    break
    
    return best_w, best_score

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    deadline = time.time() + 600.0
    safety_margin = 30.0
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # Strategy:
    # 1. Start with the known optimal 2-point solution.
    # 2. Expand to 3, 4, 5 points using geometric/log-spaced supports centered around the optimal ratio.
    # 3. Use coordinate ascent to optimize weights.
    
    # Analytical 2-point optimum
    # Let support be {0, R}. Weights {p, 1-p}.
    # X in {0, R}.
    # P(X1+X2+X3 < 2X4)
    # If X4=0 (prob 1-p), 2X4=0. Sum < 0 is impossible (sum >= 0). Prob 0.
    # If X4=R (prob p), 2X4=2R.
    #   Sum < 2R.
    #   Max sum is 3R.
    #   Sum = k*R where k is number of R's in X1,X2,X3.
    #   k=0: Sum=0 < 2R. Prob (1-p)^3
    #   k=1: Sum=R < 2R. Prob 3p(1-p)^2
    #   k=2: Sum=2R < 2R. False.
    #   k=3: Sum=3R < 2R. False.
    #   So Prob = (1-p)^3 + 3p(1-p)^2 = (1-p)^2 (1-p+3p) = (1-p)^2 (1+2p)
    # Total P = p * (1-p)^2 * (1+2p)
    # Maximize f(p) = p(1-p)^2(1+2p)
    # f'(p) = 0 -> p = (1+sqrt(33))/16 approx 0.4215?
    # Let's check: (1+sqrt(33)) is approx 6.745. /16 = 0.4215.
    # Let's verify derivative.
    # f(p) = p(1 - 2p + p^2)(1+2p) = p(1 + 2p - 2p - 4p^2 + p^2 + 2p^3) = p(1 - 3p^2 + 2p^3) = p - 3p^3 + 2p^4
    # f'(p) = 1 - 9p^2 + 8p^3
    # 8p^3 - 9p^2 + 1 = 0
    # p=1 is a root: 8-9+1=0.
    # (p-1)(8p^2 - p - 1) = 0
    # 8p^2 - p - 1 = 0 -> p = (1 + sqrt(1+32))/16 = (1+sqrt(33))/16.
    # Yes.
    
    p_opt = (1 + np.sqrt(33)) / 16.0
    R_opt = 1.0 # Scale doesn't matter for 2-point
    
    s2 = np.array([0.0, R_opt])
    w2 = np.array([1.0 - p_opt, p_opt])
    score2 = evaluate_measure(s2, w2)
    
    if score2 > best_score:
        best_score = score2
        best_support = s2.copy()
        best_weights = w2.copy()
        
    # Now try 3, 4, 5 points
    # Support structure: {0, a, b, R} or similar
    # We want X4 to be large. So put mass on R.
    # We want X1+X2+X3 to be small. So put mass on 0 and small values.
    # Intermediate values might help if they are small enough to keep sum < 2R.
    
    n_points_list = [3, 4, 5, 6]
    
    for n in n_points_list:
        if time.time() + 30.0 > deadline - safety_margin:
            break
            
        # Generate candidate supports
        # 1. Log-spaced between 0 and R
        # Avoid 0 in log space, use small epsilon
        eps = 1e-6
        if n == 3:
            # {0, mid, R}
            s_cand = np.array([0.0, 0.5, 1.0])
            w_init = np.array([0.4, 0.1, 0.5])
        elif n == 4:
            # {0, small, mid, R}
            s_cand = np.array([0.0, 0.1, 0.5, 1.0])
            w_init = np.array([0.3, 0.1, 0.1, 0.5])
        elif n == 5:
            s_cand = np.array([0.0, 0.05, 0.2, 0.5, 1.0])
            w_init = np.array([0.3, 0.1, 0.1, 0.1, 0.4])
        else:
            # Geometric spacing
            s_cand = np.geomspace(0.01, 1.0, n-1)
            s_cand = np.concatenate([[0.0], s_cand])
            w_init = np.ones(n) / n
            w_init[0] = 0.3
            w_init[-1] = 0.4
            w_init = w_init / np.sum(w_init)
            
        # Optimize weights
        time_left = deadline - safety_margin - time.time() - 5.0
        if time_left < 10.0:
            continue
            
        w_opt, score_opt = coordinate_ascent_weights(s_cand, w_init, time_left, seed=n)
        
        if score_opt > best_score:
            best_score = score_opt
            best_support = s_cand.copy()
            best_weights = w_opt.copy()
            
        # Also try a different support structure: 
        # Concentrate mass on 0 and R, with very small mass in between
        # This is essentially the 2-point solution with noise.
        # We can try to optimize the position of intermediate points.
        
    # Final validation and cleanup
    if best_support is None:
        best_support = s2
        best_weights = w2
        
    best_support = np.asarray(best_support, dtype=np.float64).flatten()
    best_weights = np.asarray(best_weights, dtype=np.float64).flatten()
    
    # Ensure non-negative
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Normalize weights
    w_sum = np.sum(best_weights)
    if w_sum > 1e-9:
        best_weights = best_weights / w_sum
    else:
        n = len(best_support)
        best_weights = np.ones(n) / n
        
    return best_support, best_weights

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print(f"Score: {evaluate_measure(s, w)}")
    print(f"n = {len(s)}")
