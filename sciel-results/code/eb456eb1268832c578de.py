import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _calc_P_fast(support, weights):
    """
    Calculate P[X1 + X2 + X3 < 2*X4] for discrete distribution.
    Uses sorted pairwise sums for speed.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    n = len(s)
    if n == 0:
        return 0.0
    
    # Normalize weights
    w_sum = np.sum(w)
    if w_sum <= 1e-12:
        return 0.0
    w = w / w_sum
    
    # Precompute all pairwise sums and their weights
    S = s[:, None] + s[None, :]
    W = w[:, None] * w[None, :]
    
    # Flatten and sort by sum
    S_flat = S.flatten()
    W_flat = W.flatten()
    
    # Sort by pairwise sum
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    W_sorted = W_flat[sort_idx]
    
    # Cumulative weights for fast threshold queries
    W_cum = np.cumsum(W_sorted)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            # Find number of pairs with sum < threshold using searchsorted
            idx = np.searchsorted(S_sorted, threshold, side='left')
            if idx > 0:
                sum_w = W_cum[idx - 1]
            else:
                sum_w = 0.0
            P += w[k] * w[l] * sum_w
    
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    def evaluate(s, w):
        """Evaluate P and update best if improved."""
        nonlocal best_score, best_support, best_weights
        s = np.asarray(s, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        
        # Ensure validity
        if len(s) != len(w) or len(s) == 0:
            return 0.0
        if np.any(s < -1e-9) or np.any(w < -1e-9):
            return 0.0
        w_sum = np.sum(w)
        if w_sum <= 1e-9:
            return 0.0
        if np.any(np.isnan(s)) or np.any(np.isnan(w)) or np.any(np.isinf(s)) or np.any(np.isinf(w)):
            return 0.0
            
        w = w / w_sum
        p_val = _calc_P_fast(s, w)
        
        if p_val > best_score:
            best_score = p_val
            best_support = s.copy()
            best_weights = w.copy()
        return p_val
    
    # Try different n values
    for n in [2, 3, 4, 5, 6]:
        if time.time() > deadline:
            break
            
        # Phase 1: Differential evolution for global search
        if time.time() < deadline - 300:  # Reserve time for local refinement
            def de_objective(params):
                s_params = params[:n]
                w_logits = params[n:]
                
                # Support must be non-negative
                s = np.abs(s_params)
                
                # Weights via softmax
                w_logits = w_logits - np.max(w_logits)
                w = np.exp(w_logits)
                w_sum = np.sum(w)
                if w_sum <= 1e-12:
                    return 1.0
                w = w / w_sum
                
                return -_calc_P_fast(s, w)
            
            # Define bounds
            s_bounds = [(0, 5)] * n
            w_bounds = [(-5, 5)] * n  # Logit bounds
            bounds = s_bounds + w_bounds
            
            # Initial population
            pop_size = min(50, max(20, 600 // (n * n)))
            
            try:
                de_result = differential_evolution(
                    de_objective, bounds,
                    seed=42,
                    maxiter=100,
                    popsize=pop_size,
                    tol=1e-6,
                    polish=False,
                    update_interval=1,
                    mutation=(0.5, 1.5),
                    recombination=0.8
                )
                
                # Evaluate the DE result
                s_de = np.abs(de_result.x[:n])
                w_logits = de_result.x[n:]
                w_de = np.exp(w_logits - np.max(w_logits))
                w_de = w_de / np.sum(w_de)
                evaluate(s_de, w_de)
                
            except Exception:
                pass
        
        # Phase 2: Local refinement with Nelder-Mead from multiple starts
        if time.time() < deadline - 60:
            def nm_objective(params):
                s_params = params[:n]
                w_logits = params[n:]
                
                s = np.abs(s_params)
                
                w_logits = w_logits - np.max(w_logits)
                w = np.exp(w_logits)
                w_sum = np.sum(w)
                if w_sum <= 1e-12:
                    return 1.0
                w = w / w_sum
                
                return -_calc_P_fast(s, w)
            
            # Generate diverse initial points
            starts = []
            
            # Start 1: Uniform
            s_init = np.linspace(0, 1, n)
            w_init = np.zeros(n)
            starts.append(np.concatenate([s_init, w_init]))
            
            # Start 2: Two-point concentrated
            s_init2 = np.zeros(n)
            s_init2[-1] = 1.0
            w_init2 = np.zeros(n)
            w_init2[0] = 0
            w_init2[-1] = 0
            starts.append(np.concatenate([s_init2, w_init2]))
            
            # Random starts
            num_random = min(8, max(3, 300 // (n * n)))
            for _ in range(num_random):
                if time.time() > deadline - 30:
                    break
                s_rand = np.sort(np.random.uniform(0, 2, n))
                w_rand = np.random.uniform(-2, 2, n)
                starts.append(np.concatenate([s_rand, w_rand]))
            
            for p_init in starts:
                if time.time() > deadline - 30:
                    break
                try:
                    res = minimize(nm_objective, p_init, method='Nelder-Mead',
                                   options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9})
                    
                    s_opt = np.abs(res.x[:n])
                    w_logits = res.x[n:]
                    w_opt = np.exp(w_logits - np.max(w_logits))
                    w_opt = w_opt / np.sum(w_opt)
                    
                    evaluate(s_opt, w_opt)
                    
                except Exception:
                    pass
    
    # If optimization failed or found nothing, fall back to 2-point analytic solution
    if best_support is None or best_score < 0.2:
        p = (15 - np.sqrt(33)) / 16.0
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([p, 1-p])
    
    # Ensure validity
    best_support = np.asarray(best_support, dtype=np.float64)
    best_weights = np.asarray(best_weights, dtype=np.float64)
    
    # Flatten
    best_support = best_support.flatten()
    best_weights = best_weights.flatten()
    
    return best_support, best_weights
