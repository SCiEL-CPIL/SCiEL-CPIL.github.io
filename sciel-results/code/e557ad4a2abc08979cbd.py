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

def precompute_influence(s):
    """
    Precompute the binary matrix M[i, j] which is 1 if s_i + s_j < 2*s_k - s_l?
    No, the objective is P = sum_{k,l} p_k p_l Q_{kl} where Q_{kl} = sum_{i,j} p_i p_j I(s_i+s_j < 2s_l - s_k).
    
    We can rewrite P as a quadratic form in p?
    P = sum_{k,l,i,j} p_k p_l p_i p_j I(s_i+s_j < 2s_l - s_k).
    
    This is a 4th order tensor contraction.
    However, we can optimize weights p by coordinate ascent.
    Fix all p_m for m != t. The objective is linear in p_t.
    P(p_t) = p_t * (sum_{k,l,i,j: t not in {k,l,i,j}} ...) + terms involving p_t.
    
    Actually, since P is homogeneous of degree 4 in p (before normalization) or degree 0 (after normalization),
    and we are constrained to sum(p)=1, p>=0.
    
    Let's just use a fast evaluator for weight updates.
    The expensive part is the searchsorted. We can precompute the indices for a fixed support s.
    """
    n = len(s)
    ss = s[:, None] + s[None, :]
    ss_flat = ss.flatten()
    sort_idx = np.argsort(ss_flat)
    ss_sorted = ss_flat[sort_idx]
    
    # thresholds for all (k, l)
    thresholds = 2.0 * s[None, :] - s[:, None]
    idx = np.searchsorted(ss_sorted, thresholds, side='left')
    
    return idx, ss_sorted

def eval_weights_fast(p, idx, ss_sorted, s):
    """
    Evaluate P given probabilities p, precomputed idx, and support s.
    """
    n = len(s)
    pp = p[:, None] * p[None, :]
    pp_flat = pp.flatten()
    
    # We need cumulative sum of pp_flat sorted by ss_flat
    # But ss_flat is fixed for fixed s.
    # So we just need to sort pp_flat according to the same order as ss_flat.
    # sort_idx was used to sort ss_flat.
    # So pp_sorted = pp_flat[sort_idx]
    # But we didn't return sort_idx. Let's return it or recompute.
    # Recomputing sort_idx is O(n^2 log n^2) = O(n^2 log n).
    # For n=20, n^2=400, log 400 ~ 9. 3600 ops. Very fast.
    
    # Let's just do the full evaluation but vectorized, which is fast enough for n<=30.
    # The previous code was already doing this.
    # The bottleneck in the original code was the *optimization* loop, not the evaluation.
    # Evaluation for n=20 is ~1ms.
    # So we can afford ~1000 evaluations per second.
    # In 600s, we can do 600,000 evaluations.
    
    # Let's stick to the vectorized evaluation which is simple and robust.
    pass

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    deadline = time.time() + 600.0
    safety_margin = 30.0
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # Strategy:
    # 1. Define a parametric family of supports: geometric sequences s_i = A * r^i.
    #    Or log-spaced: s_i = exp(A + i * B).
    # 2. For a fixed n, optimize (A, B) and weights w.
    #    Use coordinate ascent on weights for fixed support.
    #    Use random restarts / grid search for support parameters.
    
    # Let's try n=15 to n=25.
    # For each n, try a set of candidate supports generated from log-spaced grids with different start/end/ratio.
    # Then optimize weights.
    
    n_candidates = [12, 16, 20, 24]
    
    for n in n_candidates:
        if time.time() + 30.0 > deadline - safety_margin:
            break
        
        # Generate candidate supports
        # Log-spaced: s_i = s_min * (s_max/s_min)^(i/(n-1))
        # Let's vary s_min and s_max.
        # The inequality is X1+X2+X3 < 2*X4.
        # If distribution is concentrated on small values, X4 is small, RHS is small.
        # If distribution is heavy-tailed, X4 can be large.
        
        # Let's try supports that cover a wide range.
        # s_min in [0.1, 1, 10], s_max in [10, 100, 1000, 10000]
        
        s_mins = [0.0, 0.1, 1.0]
        s_maxs = [10.0, 100.0, 1000.0, 10000.0]
        
        for s_min in s_mins:
            for s_max in s_maxs:
                if time.time() + 10.0 > deadline - safety_margin:
                    break
                
                if s_max <= s_min:
                    continue
                
                # Geometric progression
                if s_min == 0:
                    # Can't do geometric with 0. Use linear or shifted geometric.
                    # Let's use s_i = i * (s_max / (n-1)) for i=0..n-1
                    s = np.linspace(0, s_max, n)
                else:
                    ratio = (s_max / s_min) ** (1.0 / (n - 1))
                    s = s_min * (ratio ** np.arange(n))
                
                # Optimize weights for this support
                # Start with uniform weights
                w = np.ones(n) / n
                
                # Coordinate ascent on weights
                # Max iterations
                max_iter = 500
                converged = False
                
                for it in range(max_iter):
                    if time.time() + 5.0 > deadline - safety_margin:
                        break
                    
                    old_w = w.copy()
                    improved = False
                    
                    for i in range(n):
                        # Move weight from i to others?
                        # Or adjust w_i while keeping sum 1.
                        # Simple move: shift delta from i to j.
                        # Better: solve for optimal w_i given others fixed?
                        # P is linear in w_i if we treat normalization separately?
                        # No, P is multilinear.
                        
                        # Let's just do a simple random pair swap hill climb, but faster.
                        # Or grid search on w_i?
                        
                        # Let's try: for each i, try a few values for w_i, renormalize others.
                        # This is O(n * K * eval_cost).
                        
                        # Actually, let's just use a simple random restart hill climbing with many steps,
                        # but we have time.
                        
                        pass # Placeholder for inner loop logic in the actual implementation below
                    
                    # Since the inner loop above was a placeholder, let's implement a proper weight optimizer
                    # that runs for a fixed time budget per support candidate.
                    
                    break # Break out of placeholder loop to use the dedicated optimizer below
        
        # Dedicated weight optimizer for the best support found in the grid?
        # No, let's run the weight optimizer for EACH candidate support, but with a strict time limit.
        
        # Re-structure:
        # 1. Generate list of candidate supports.
        # 2. For each, run weight optimization for T seconds.
        # 3. Keep the best.
        
        pass # The above loop was messy. Let's rewrite cleanly.

    # Clean implementation:
    
    def optimize_weights(s, time_limit, seed=42):
        n = len(s)
        if n == 0:
            return np.array([1.0]), 0.0
        
        best_w = np.ones(n) / n
        best_score = evaluate_measure(s, best_w)
        
        start_time = time.time()
        rng = np.random.default_rng(seed)
        
        # Use a faster local search:
        # Random pair exchange with adaptive step size.
        
        step_size = 0.1
        stuck_count = 0
        
        while time.time() - start_time < time_limit:
            i1 = rng.integers(0, n)
            i2 = rng.integers(0, n)
            if i1 == i2:
                continue
                
            if best_w[i1] < 1e-6:
                continue
                
            # Random direction and magnitude
            max_move = best_w[i1] * 0.5
            move = max_move * (1 if rng.random() < 0.5 else -1) * step_size
            
            if move > 0 and best_w[i1] < move:
                move = best_w[i1] * 0.5
            if move < 0 and best_w[i2] < -move:
                move = best_w[i2] * 0.5
                
            if abs(move) < 1e-9:
                continue
                
            w_trial = best_w.copy()
            w_trial[i1] -= move
            w_trial[i2] += move
            
            if np.any(w_trial < -1e-9):
                continue
                
            score_trial = evaluate_measure(s, w_trial)
            
            if score_trial > best_score + 1e-12:
                best_score = score_trial
                best_w = w_trial
                stuck_count = 0
                step_size *= 0.95
            else:
                stuck_count += 1
                step_size *= 1.05
                if step_size > 0.5:
                    step_size = 0.5
                if step_size < 0.001:
                    step_size = 0.001
                    if stuck_count > 50:
                        # Restart local search
                        best_w = rng.dirichlet(np.ones(n) * 0.5)
                        best_score = evaluate_measure(s, best_w)
                        step_size = 0.1
                        stuck_count = 0
        
        return best_w, best_score

    # Generate candidates
    candidates = []
    
    # 1. Geometric supports
    for n in [10, 15, 20, 25]:
        for s_min in [0.0, 0.1, 1.0]:
            for s_max in [10.0, 100.0, 1000.0]:
                if s_max <= s_min:
                    continue
                if s_min == 0:
                    s = np.linspace(0, s_max, n)
                else:
                    ratio = (s_max / s_min) ** (1.0 / (n - 1))
                    s = s_min * (ratio ** np.arange(n))
                candidates.append(s)
                
    # 2. Log-uniform supports (points are log-spaced)
    for n in [15, 20, 25]:
        for log_min in [-2, -1, 0]:
            for log_max in [2, 3, 4]:
                if log_max <= log_min:
                    continue
                log_s = np.linspace(log_min, log_max, n)
                s = np.exp(log_s)
                candidates.append(s)
                
    # 3. Bimodal supports
    for n in [20, 25]:
        # Half points near 0, half points large
        n1 = n // 2
        n2 = n - n1
        s1 = np.linspace(0, 1, n1)
        s2 = np.linspace(10, 100, n2)
        s = np.concatenate([s1, s2])
        candidates.append(s)
        
        s1 = np.linspace(0, 5, n1)
        s2 = np.linspace(100, 1000, n2)
        s = np.concatenate([s1, s2])
        candidates.append(s)

    # Evaluate candidates
    for idx, s in enumerate(candidates):
        if time.time() + 15.0 > deadline - safety_margin:
            break
            
        # Allocate time based on remaining budget
        remaining = deadline - safety_margin - time.time()
        # Use a fraction of remaining time, decreasing as we go
        time_per_candidate = max(5.0, remaining / (len(candidates) - idx))
        time_per_candidate = min(time_per_candidate, 30.0) # Cap at 30s per candidate
        
        w, score = optimize_weights(s, time_per_candidate, seed=idx)
        
        if score > best_score:
            best_score = score
            best_support = s.copy()
            best_weights = w.copy()
            
    # If no candidate was good enough, try a generic fallback
    if best_support is None:
        n = 15
        s = np.linspace(0, 100, n)
        w = np.ones(n) / n
        best_support = s
        best_weights = w
        best_score = evaluate_measure(s, w)

    # Final validation and cleanup
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
