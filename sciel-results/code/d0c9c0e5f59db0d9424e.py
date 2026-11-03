import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _score_smooth(h, tau=0.1):
    """
    Smoothed version of the score using log-sum-exp for the max.
    h is expected to be normalized such that sum(h)=1 for stability, 
    but we handle normalization internally for robustness.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2)
    
    h_conv = np.convolve(h_norm, h_norm, mode='full')
    # Scale by DELTA as in raw score
    c = DELTA * h_conv
    
    # Soft max approximation
    # max(c) ~ tau * log(sum(exp(c/tau)))
    # To avoid overflow, subtract max
    c_max = np.max(c)
    log_sum_exp = c_max + tau * np.log(np.sum(np.exp((c - c_max) / tau)))
    
    if log_sum_exp == 0:
        return 0.0
        
    return l2_sq / log_sum_exp

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Helper: optimize amplitudes for a fixed support
    def optimize_amplitudes(support_indices):
        k = len(support_indices)
        if k < 2:
            return None, 0.0
        
        def neg_obj(x):
            a = np.exp(x)
            a_sum = np.sum(a)
            if a_sum == 0:
                return 0.0
            a = a / a_sum
            h = np.zeros(N)
            h[support_indices] = a
            s = _score_raw(h)
            if s <= 0:
                return 0.0
            return -s
        
        best_local_h = None
        best_local_s = 0.0
        
        # Parametric exponential skew
        def neg_obj_alpha(alpha):
            a = np.exp(alpha * np.arange(k))
            a_sum = np.sum(a)
            if a_sum == 0:
                return 0.0
            a = a / a_sum
            h = np.zeros(N)
            h[support_indices] = a
            s = _score_raw(h)
            if s <= 0:
                return 0.0
            return -s
        
        best_alpha = 0.0
        best_alpha_s = 0.0
        for alpha in np.linspace(-5, 5, 30):
            s_val = -neg_obj_alpha(alpha)
            if s_val > best_alpha_s:
                best_alpha_s = s_val
                best_alpha = alpha
        
        try:
            res_alpha = minimize(neg_obj_alpha, best_alpha, method='Nelder-Mead', 
                                 options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-10})
            if -res_alpha.fun > best_alpha_s:
                best_alpha_s = -res_alpha.fun
                best_alpha = res_alpha.x[0]
        except Exception:
            pass
        
        a = np.exp(best_alpha * np.arange(k))
        a = a / np.sum(a)
        h_candidate = np.zeros(N)
        h_candidate[support_indices] = a
        s_candidate = _score_raw(h_candidate)
        
        if s_candidate > best_local_s:
            best_local_s = s_candidate
            best_local_h = h_candidate.copy()
            
        # Multivariate refinement
        if k <= 12 and time.time() < deadline - 5:
            x0 = best_alpha * np.arange(k)
            try:
                res = minimize(neg_obj, x0, method='L-BFGS-B', 
                              options={'maxiter': 300, 'ftol': 1e-12})
                if res.fun < 0:
                    a = np.exp(res.x)
                    a = a / np.sum(a)
                    h_test = np.zeros(N)
                    h_test[support_indices] = a
                    s_test = _score_raw(h_test)
                    if s_test > best_local_s:
                        best_local_s = s_test
                        best_local_h = h_test.copy()
            except Exception:
                pass
        
        return best_local_h, best_local_s

    # Phase 1: Smooth Global Optimization via Soft-Max
    # We parameterize h_i = exp(x_i). We optimize x.
    # Objective: -_score_smooth(exp(x))
    
    def smooth_neg_obj(x):
        # x are log-amplitudes
        # Shift x to prevent underflow/overflow
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        h_sum = np.sum(h)
        if h_sum == 0:
            return 0.0
        h = h / h_sum
        s = _score_smooth(h, tau=0.1)
        if s <= 0:
            return 0.0
        return -s

    # Initializations for smooth optimization
    inits = []
    
    # 1. Uniform
    inits.append(np.zeros(N))
    
    # 2. Random sparse (random support of size 5-15)
    np.random.seed(42)
    for _ in range(20):
        k = np.random.randint(5, 16)
        support = np.random.choice(N, k, replace=False)
        x0 = np.full(N, -10.0) # Very small amplitudes for inactive
        x0[support] = np.random.uniform(0, 2, k)
        inits.append(x0)
        
    # 3. Sidon-like initializations (greedy)
    def generate_sidon_support(size):
        points = [0]
        for _ in range(size - 1):
            best_candidate = -1
            best_score_candidate = -1
            current_diffs = []
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    current_diffs.append(abs(points[i] - points[j]))
            for c in range(N):
                if c in points:
                    continue
                new_diffs = current_diffs.copy()
                min_dist = float('inf')
                for p in points:
                    d = abs(c - p)
                    new_diffs.append(d)
                    if d < min_dist:
                        min_dist = d
                unique_diffs = len(set(new_diffs))
                total_diffs = len(new_diffs)
                collision_penalty = total_diffs - unique_diffs
                score = min_dist * 10 - collision_penalty
                if score > best_score_candidate:
                    best_score_candidate = score
                    best_candidate = c
            if best_candidate != -1:
                points.append(best_candidate)
            else:
                break
        return sorted(points)
        
    for size in [5, 8, 12]:
        supp = generate_sidon_support(size)
        if len(supp) == size:
            x0 = np.full(N, -10.0)
            x0[supp] = 0.0
            inits.append(x0)

    # Run smooth optimization for each init
    for x0 in inits:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(smooth_neg_obj, x0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12})
            
            # Reconstruct h
            x_opt = res.x
            x_shifted = x_opt - np.max(x_opt)
            h_opt = np.exp(x_shifted)
            h_opt = h_opt / np.sum(h_opt)
            
            # Check raw score
            s_opt = _score_raw(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
                
            # Extract active support (top K amplitudes) and refine
            # Threshold: amplitudes > 1% of max
            max_amp = np.max(h_opt)
            threshold = max_amp * 0.01
            active_support = np.where(h_opt > threshold)[0]
            
            if len(active_support) >= 2:
                h_refined, s_refined = optimize_amplitudes(list(active_support))
                if h_refined is not None and s_refined > best_score:
                    best_score = s_refined
                    best_h = h_refined.copy()
                    
        except Exception:
            pass

    # Phase 2: Local Refinement of Best Support
    if time.time() < deadline - 10:
        active = np.where(best_h > 1e-6)[0]
        active_list = list(active)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(len(active_list)):
                if time.time() > deadline:
                    break
                # Try swapping with a few random candidates instead of all N
                # to save time, or all N if support is small
                candidates = all_indices = np.arange(N)
                for j in candidates:
                    if j in active_list:
                        continue
                    new_support = active_list.copy()
                    new_support[i] = j
                    new_support = sorted(new_support)
                    h_candidate, s_candidate = optimize_amplitudes(new_support)
                    if h_candidate is not None and s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate.copy()
                        active_list = new_support
                        improved = True
                        break # Restart
                if improved:
                    break

    return list(best_h)
