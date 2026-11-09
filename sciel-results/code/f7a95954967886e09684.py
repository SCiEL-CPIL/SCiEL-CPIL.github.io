import numpy as np
import time
import sys
from scipy.optimize import differential_evolution

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

def precompute_support(s):
    """Precompute sorted pair sums for a given support."""
    ss = s[:, None] + s[None, :]
    ss_flat = ss.flatten()
    sort_idx = np.argsort(ss_flat)
    ss_sorted = ss_flat[sort_idx]
    return ss_sorted

def eval_with_precomp(s, w, ss_sorted):
    """Evaluate using precomputed sorted pair sums."""
    n = len(s)
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 0.0
    p = w / w_sum
    p = np.maximum(p, 0.0)
    s = np.maximum(s, 0.0)
    p_sum = np.sum(p)
    if p_sum <= 1e-9:
        return 0.0
    p = p / p_sum
    
    pp = p[:, None] * p[None, :]
    pp_flat = pp.flatten()
    
    # We need to re-sort pp_flat according to the same order as ss_sorted
    # But ss_sorted was computed from s, and s might have changed slightly?
    # No, in this function s is fixed for the precomputation.
    # However, the order of pp_flat must match the order of ss_sorted.
    # ss_sorted corresponds to the flattened ss in row-major order, sorted.
    # pp_flat is in row-major order of pp.
    # So we need to sort pp_flat by the indices that sorted ss_flat.
    # Actually, we can just recompute the sort indices? No, that's slow.
    # Better: precompute the sort indices once.
    # Let's modify precompute_support to return sort_idx as well.
    pass # Placeholder

def precompute_support_full(s):
    """Precompute sorted pair sums and sort indices."""
    ss = s[:, None] + s[None, :]
    ss_flat = ss.flatten()
    sort_idx = np.argsort(ss_flat)
    ss_sorted = ss_flat[sort_idx]
    return ss_sorted, sort_idx

def eval_with_precomp_full(s, w, ss_sorted, sort_idx):
    """Evaluate using precomputed sorted pair sums and sort indices."""
    n = len(s)
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 0.0
    p = w / w_sum
    p = np.maximum(p, 0.0)
    s = np.maximum(s, 0.0)
    p_sum = np.sum(p)
    if p_sum <= 1e-9:
        return 0.0
    p = p / p_sum
    
    pp = p[:, None] * p[None, :]
    pp_flat = pp.flatten()
    pp_sorted = pp_flat[sort_idx]
    
    C = np.cumsum(pp_sorted)
    
    thresholds = 2.0 * s[None, :] - s[:, None]
    idx = np.searchsorted(ss_sorted, thresholds, side='left')
    prob_inner = np.zeros_like(idx, dtype=np.float64)
    mask = idx > 0
    prob_inner[mask] = C[idx[mask] - 1]
    
    P = np.sum(p[:, None] * p[None, :] * prob_inner)
    return P

def optimize_jointly(s_init, w_init, time_limit, seed=42):
    """
    Jointly optimize support and weights using random restarts and local search.
    """
    n = len(s_init)
    best_s = s_init.copy()
    best_w = w_init.copy()
    best_score = evaluate_measure(best_s, best_w)
    
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    # Precompute for initial
    ss_sorted, sort_idx = precompute_support_full(best_s)
    
    max_restarts = 30
    max_steps = 5000
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize
        if restart == 0:
            s = best_s.copy()
            w = best_w.copy()
        else:
            # Perturb support
            s = best_s + rng.normal(0, 0.1, n)
            s = np.maximum(s, 0)
            s = np.sort(s)
            # Ensure strictly increasing
            for i in range(1, n):
                if s[i] <= s[i-1]:
                    s[i] = s[i-1] * 1.01 + 1e-6
            
            # Perturb weights
            w = best_w + rng.normal(0, 0.05, n)
            w = np.maximum(w, 0)
            w_sum = np.sum(w)
            if w_sum > 1e-9:
                w = w / w_sum
            else:
                w = np.ones(n) / n
        
        current_score = evaluate_measure(s, w)
        
        if current_score > best_score:
            best_score = current_score
            best_s = s.copy()
            best_w = w.copy()
        
        # Local Search
        step_size_s = 0.1
        step_size_w = 0.05
        stuck = 0
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                return best_s, best_w, best_score
            
            # Choose whether to perturb support or weights
            if rng.random() < 0.5:
                # Perturb support
                i = rng.integers(0, n)
                delta = rng.normal(0, step_size_s)
                s_new = s.copy()
                s_new[i] += delta
                s_new = np.maximum(s_new, 0)
                s_new = np.sort(s_new)
                # Ensure strictly increasing
                valid = True
                for j in range(1, n):
                    if s_new[j] <= s_new[j-1]:
                        s_new[j] = s_new[j-1] * 1.01 + 1e-6
                
                score_new = evaluate_measure(s_new, w)
                
                if score_new > current_score + 1e-12:
                    s = s_new
                    current_score = score_new
                    if current_score > best_score:
                        best_score = current_score
                        best_s = s.copy()
                        best_w = w.copy()
                    stuck = 0
                    step_size_s *= 0.95
                else:
                    stuck += 1
                    step_size_s *= 1.05
                    if step_size_s > 0.5:
                        step_size_s = 0.5
                    if stuck > 50:
                        break
            else:
                # Perturb weights
                i1 = rng.integers(0, n)
                i2 = rng.integers(0, n)
                if i1 == i2:
                    continue
                
                move = rng.normal(0, step_size_w)
                w_new = w.copy()
                w_new[i1] -= move
                w_new[i2] += move
                
                if np.any(w_new < -1e-9):
                    continue
                
                w_new = np.maximum(w_new, 0)
                w_sum = np.sum(w_new)
                if w_sum <= 1e-9:
                    continue
                w_new = w_new / w_sum
                
                score_new = evaluate_measure(s, w_new)
                
                if score_new > current_score + 1e-12:
                    w = w_new
                    current_score = score_new
                    if current_score > best_score:
                        best_score = current_score
                        best_s = s.copy()
                        best_w = w.copy()
                    stuck = 0
                    step_size_w *= 0.95
                else:
                    stuck += 1
                    step_size_w *= 1.05
                    if step_size_w > 0.3:
                        step_size_w = 0.3
                    if stuck > 50:
                        break

    return best_s, best_w, best_score

def generate_support_candidates(n_points):
    """Generate diverse initial support candidates."""
    candidates = []
    
    # Geometric
    r = np.random.default_rng(1)
    for _ in range(5):
        ratio = r.uniform(1.5, 4.0)
        s = np.array([0.0] + [ratio ** i for i in range(1, n_points)])
        s = s / s[-1] * r.uniform(1, 10)
        candidates.append(s)
    
    # Log-normal mixture
    for _ in range(5):
        mu1 = r.uniform(-1, 3)
        sigma1 = r.uniform(0.5, 2)
        mu2 = r.uniform(0, 5)
        sigma2 = r.uniform(0.5, 2)
        w1 = r.uniform(0.2, 0.8)
        
        u = np.linspace(0.01, 0.99, n_points)
        s = []
        for ui in u:
            # Bisection for inverse CDF
            lo, hi = 1e-6, 1e6
            for _ in range(50):
                mid = np.sqrt(lo * hi)
                cdf = w1 * 0.5 * (1 + np.erf((np.log(mid) - mu1) / (sigma1 * np.sqrt(2)))) + \
                      (1 - w1) * 0.5 * (1 + np.erf((np.log(mid) - mu2) / (sigma2 * np.sqrt(2))))
                if cdf < ui:
                    lo = mid
                else:
                    hi = mid
            s.append(np.sqrt(lo * hi))
        s = np.array(s)
        candidates.append(s)
    
    # Uniform with noise
    for _ in range(5):
        s = np.sort(r.uniform(0, 10, n_points))
        candidates.append(s)
    
    return candidates

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    deadline = time.time() + 600.0
    safety_margin = 30.0
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # Try different numbers of support points
    n_points_list = [8, 12, 16, 20, 24]
    
    for n_points in n_points_list:
        if time.time() + 60.0 > deadline - safety_margin:
            break
        
        # Generate candidates
        candidates = generate_support_candidates(n_points)
        
        for s_init in candidates:
            if time.time() + 30.0 > deadline - safety_margin:
                break
            
            # Initialize weights
            w_init = np.ones(n_points) / n_points
            
            # Allocate time for optimization
            time_budget = min(60.0, deadline - safety_margin - time.time() - 10.0)
            if time_budget < 10.0:
                break
            
            s_opt, w_opt, score_opt = optimize_jointly(s_init, w_init, time_budget, seed=int(time.time() * 1000) % 10000)
            
            if score_opt > best_score:
                best_score = score_opt
                best_support = s_opt.copy()
                best_weights = w_opt.copy()
        
        # Refine the best found so far
        if best_support is not None:
            remaining = deadline - safety_margin - time.time() - 5.0
            if remaining > 20.0:
                s_ref, w_ref, score_ref = optimize_jointly(best_support, best_weights, remaining, seed=12345)
                if score_ref > best_score:
                    best_score = score_ref
                    best_support = s_ref.copy()
                    best_weights = w_ref.copy()
    
    # Fallback
    if best_support is None:
        n = 10
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n) / n
    
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
