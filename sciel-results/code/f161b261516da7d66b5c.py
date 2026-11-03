import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    # Continuous autoconvolution of step function f(t) = sum h_i 1_{[x_i, x_{i+1})}(t)
    # The peak is proportional to the max of the discrete convolution of h with itself.
    # Specifically, max(f*f) = DELTA * max(conv(h,h))
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _smooth_score(h, alpha=10.0):
    """
    Smooth surrogate for the score.
    Replaces max(conv(h,h)) with a soft-max approximation using log-sum-exp.
    This makes the objective differentiable almost everywhere, helping SLSQP.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    
    h_conv = np.convolve(h, h, mode='full')
    
    # Soft max of h_conv
    # max(x) ≈ (1/alpha) * log(sum(exp(alpha * x)))
    # To avoid overflow, shift by max first
    m = np.max(h_conv)
    if m == 0:
        return 0.0
    # log-sum-exp
    log_sum_exp = m + np.log(np.sum(np.exp(alpha * (h_conv - m))))
    soft_max = (1.0 / alpha) * log_sum_exp
    
    # As alpha -> inf, soft_max -> max
    # We want to maximize l2_sq / (l1 * soft_max)
    # Note: soft_max >= max, so this is a lower bound on the true score.
    # Maximizing this lower bound is a valid strategy.
    
    return (l2_sq / (l1 * soft_max))

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin (limit is 120s)
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # 1. Uniform
    h_init = np.ones(N)
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # 2. Gaussian
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N/2, 10):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # 3. Triangular
    for c in np.linspace(5, N-5, 10):
        for w in np.linspace(1, N, 10):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()
    
    # 4. Random sparse initialization (Sidon-like intuition)
    # Sparse vectors tend to have smaller autoconvolution peaks relative to L2 norm
    # if the non-zeros are well-separated.
    
    candidates = []
    
    # Generate diverse initializations
    # A. Random Sparse with min distance
    num_sparse_inits = 500
    for _ in range(num_sparse_inits):
        if time.time() > deadline - 30: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(5, 15)
        # Try to pick indices with some separation
        indices = []
        attempts = 0
        while len(indices) < num_peaks and attempts < 100:
            idx = np.random.randint(N)
            if all(abs(idx - i) > 2 for i in indices):
                indices.append(idx)
            attempts += 1
        if len(indices) == 0:
            indices = np.random.choice(N, num_peaks, replace=False).tolist()
        
        h_rand[indices] = 1.0
        # Add small noise to break symmetries
        h_rand += np.random.rand(N) * 0.1
        h_rand = np.maximum(h_rand, 0)
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()
        candidates.append((s, h_rand.copy()))

    # B. Greedy Sidon-like construction
    # Minimize the maximum multiplicity of differences
    for _ in range(10):
        if time.time() > deadline - 30: break
        h_sidon = np.zeros(N)
        num_peaks = np.random.randint(8, 12)
        indices = []
        for _ in range(num_peaks):
            best_idx = -1
            best_max_mult = N
            for idx in range(N):
                if idx in indices: continue
                # Check differences with existing indices
                diffs = [abs(idx - i) for i in indices]
                # Count frequencies of diffs
                from collections import Counter
                counts = Counter(diffs)
                max_mult = max(counts.values()) if counts else 0
                if max_mult < best_max_mult:
                    best_max_mult = max_mult
                    best_idx = idx
            if best_idx != -1:
                indices.append(best_idx)
        
        if indices:
            h_sidon[indices] = 1.0
            h_sidon += np.random.rand(N) * 0.05
            h_sidon = np.maximum(h_sidon, 0)
            s = _score_raw(h_sidon)
            if s > best_score:
                best_score = s
                best_h = h_sidon.copy()
            candidates.append((s, h_sidon.copy()))

    # Sort candidates by score and keep top 20 for polishing
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = [h for _, h in candidates[:20]]
    
    # Ensure best_h is in top_candidates if it's very good
    if best_h not in top_candidates:
        top_candidates.append(best_h)

    # --- Optimization Phase ---
    # We maximize L2^2 / (L1 * AutoConvMax).
    # We assume the evaluator normalizes to L1=1.
    # So we optimize h subject to sum(h)=1, h>=0.
    
    def neg_obj_smooth(h, alpha=10.0):
        """
        Minimize negative smooth score.
        """
        h_pos = np.maximum(h, 1e-10)
        l2_sq = np.sum(h_pos**2)
        h_conv = np.convolve(h_pos, h_pos, mode='full')
        
        m = np.max(h_conv)
        if m < 1e-10:
            return 0.0
        log_sum_exp = m + np.log(np.sum(np.exp(alpha * (h_conv - m))))
        soft_max = (1.0 / alpha) * log_sum_exp
        
        return -(l2_sq / soft_max)

    def neg_obj_exact(h):
        """
        Minimize negative exact score.
        """
        h_pos = np.maximum(h, 1e-10)
        l1 = np.sum(h_pos)
        if l1 < 1e-10:
            return 0.0
        h_norm = h_pos / l1
        return -_score_raw(h_norm)

    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
        {'type': 'ineq', 'fun': lambda h: h} # h >= 0
    ]
    bounds = [(0.0, 1.0)] * N
    
    # Phase 1: Smooth optimization (SLSQP) on top candidates
    # This finds the basin of attraction quickly.
    polished_candidates = []
    
    for h_init in top_candidates:
        if time.time() > deadline - 10: break
        
        h_start = h_init / np.sum(h_init)
        
        try:
            res = minimize(neg_obj_smooth, h_start, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
                polished_candidates.append((s_candidate, h_candidate.copy()))
        except Exception:
            pass

    # Sort polished candidates
    polished_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Exact optimization (Nelder-Mead) on top polished candidates
    # Nelder-Mead is derivative-free and robust for the non-smooth exact objective.
    for s_init, h_init in polished_candidates[:10]:
        if time.time() > deadline - 5: break
        
        h_start = h_init / np.sum(h_init)
        
        try:
            # Use exact objective with Nelder-Mead
            # Nelder-Mead doesn't support constraints directly, so we rely on the fact that
            # the initial point is feasible and the method stays close, or we project back.
            # However, Nelder-Mead can violate bounds. 
            # To be safe, we use a penalty method or just trust it for a few steps.
            # Given the high dimensionality and constraints, SLSQP is safer for bounds.
            # Let's try SLSQP again on the EXACT objective for the final polish.
            # SLSQP with numerical gradients on the exact objective might be noisy but accurate.
            
            res = minimize(neg_obj_exact, h_start, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 200, 'ftol': 1e-12})
            if res.success:
                h_candidate = res.x
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Final check
    return list(best_h)
