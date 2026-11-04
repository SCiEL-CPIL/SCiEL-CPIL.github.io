import numpy as np
import scipy.optimize
import time
from scipy.special import logsumexp

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin from 150s limit
    
    N = 100
    target_sum = N / 2.0
    
    def get_correlations_linear(h):
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero
    
    def score_h(h):
        c_all = get_correlations_linear(h)
        return np.max(c_all)
    
    def lse_smooth_max(c, beta):
        max_c = np.max(c)
        if max_c > 100:
            return max_c
        shifted = beta * (c - max_c)
        # Clip to avoid overflow in exp
        shifted = np.clip(shifted, -500, 0)
        log_sum = np.log(np.sum(np.exp(shifted)))
        return max_c + log_sum / beta
    
    def epigraph_objective(h, t, beta, lambda_sum):
        """Epigraph formulation: minimize t + smooth_violation + sum_penalty"""
        c_all = get_correlations_linear(h)
        # Smooth max violation: sum of soft-plus of (c_i - t)
        violations = np.maximum(c_all - t, 0)
        # Use smooth approximation of sum of max(0, c_i - t)
        # soft_plus(x) = log(1 + exp(beta*x))/beta
        soft_viol = np.sum(np.logaddexp(0, beta * (c_all - t))) / beta
        sum_diff = np.sum(h) - target_sum
        pen = lambda_sum * sum_diff**2
        return t + soft_viol + pen
    
    def objective(h, beta, lambda_sum):
        c_all = get_correlations_linear(h)
        lse_val = lse_smooth_max(c_all, beta)
        sum_diff = np.sum(h) - target_sum
        pen = lambda_sum * sum_diff**2
        return lse_val + pen

    def optimize_instance(h0, beta, lambda_sum, max_iter):
        bounds = [(0, 1) for _ in range(N)]
        
        def obj(h):
            return objective(h, beta, lambda_sum)
        
        try:
            res = scipy.optimize.minimize(
                obj,
                h0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 100}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0.copy()
        
        h_opt = np.clip(h_opt, 0, 1)
        
        # Exact projection: adjust to meet sum constraint exactly
        for _ in range(50):
            current_sum = np.sum(h_opt)
            diff = target_sum - current_sum
            if abs(diff) < 1e-15:
                break
            if diff > 0:
                can_increase = h_opt < 1.0
                if np.sum(can_increase) == 0:
                    break
                h_opt[can_increase] += diff / np.sum(can_increase)
            else:
                can_decrease = h_opt > 0.0
                if np.sum(can_decrease) == 0:
                    break
                h_opt[can_decrease] += diff / np.sum(can_decrease)
            h_opt = np.clip(h_opt, 0, 1)
        
        return h_opt
    
    def optimize_epigraph(h0, beta, lambda_sum, max_iter):
        """Optimize using epigraph formulation with auxiliary variable t"""
        # Initial t estimate
        c_init = get_correlations_linear(h0)
        t0 = np.max(c_init) + 0.1
        
        # Variables: [h (N), t (1)]
        x0 = np.concatenate([h0, [t0]])
        bounds_e = [(0, 1)] * N + [(-1, 2)]
        
        def obj_e(x):
            h = x[:N]
            t = x[N]
            return epigraph_objective(h, t, beta, lambda_sum)
        
        try:
            res = scipy.optimize.minimize(
                obj_e,
                x0,
                method='L-BFGS-B',
                bounds=bounds_e,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 100}
            )
            h_opt = res.x[:N]
        except Exception:
            h_opt = h0.copy()
        
        h_opt = np.clip(h_opt, 0, 1)
        
        # Exact projection
        for _ in range(50):
            current_sum = np.sum(h_opt)
            diff = target_sum - current_sum
            if abs(diff) < 1e-15:
                break
            if diff > 0:
                can_increase = h_opt < 1.0
                if np.sum(can_increase) == 0:
                    break
                h_opt[can_increase] += diff / np.sum(can_increase)
            else:
                can_decrease = h_opt > 0.0
                if np.sum(can_decrease) == 0:
                    break
                h_opt[can_decrease] += diff / np.sum(can_decrease)
            h_opt = np.clip(h_opt, 0, 1)
        
        return h_opt

    # Phase 1: Combinatorial Search (Simulated Annealing) on Binary Sequences
    rng = np.random.default_rng(42)
    
    # Initialize with a random binary sequence with exactly N/2 ones
    h_bin = np.zeros(N)
    indices = rng.permutation(N)
    h_bin[indices[:N//2]] = 1.0
    
    best_h_bin = h_bin.copy()
    best_score_bin = score_h(h_bin)
    
    # Simulated Annealing
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.1
    
    while T > min_T and time.time() - start_time < time_limit - 40:
        ones = np.where(h_bin == 1.0)[0]
        zeros = np.where(h_bin == 0.0)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        i = ones[rng.integers(len(ones))]
        j = zeros[rng.integers(len(zeros))]
        
        h_bin[i], h_bin[j] = h_bin[j], h_bin[i]
        
        new_score = score_h(h_bin)
        delta = new_score - best_score_bin
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            best_h_bin = h_bin.copy()
            best_score_bin = new_score
        else:
            h_bin[i], h_bin[j] = h_bin[j], h_bin[i]
            
        T *= cooling_rate
        
    # Phase 2: Continuous Refinement
    h0 = best_h_bin.copy()
    best_h = h0
    best_score = best_score_bin
    
    # Try both standard and epigraph formulations
    for beta in [10.0, 30.0, 100.0]:
        if time.time() - start_time > time_limit - 15:
            break
        
        # Standard LSE approach
        h_opt1 = optimize_instance(h0, beta, 5000.0, 300)
        if np.isclose(np.sum(h_opt1), target_sum, atol=1e-10) and \
           np.all(h_opt1 >= -1e-12) and np.all(h_opt1 <= 1 + 1e-12):
            current_score1 = score_h(h_opt1)
            if current_score1 < best_score:
                best_score = current_score1
                best_h = h_opt1.copy()
                h0 = best_h.copy()
        
        # Epigraph approach
        if time.time() - start_time < time_limit - 15:
            h_opt2 = optimize_epigraph(h0, beta, 5000.0, 300)
            if np.isclose(np.sum(h_opt2), target_sum, atol=1e-10) and \
               np.all(h_opt2 >= -1e-12) and np.all(h_opt2 <= 1 + 1e-12):
                current_score2 = score_h(h_opt2)
                if current_score2 < best_score:
                    best_score = current_score2
                    best_h = h_opt2.copy()
                    h0 = best_h.copy()
    
    # Phase 3: High-precision polish pass
    if time.time() - start_time < time_limit - 10:
        for beta in [300.0, 1000.0]:
            if time.time() - start_time > time_limit - 5:
                break
            
            # Try both methods for polish
            h_opt1 = optimize_instance(best_h, beta, 50000.0, 800)
            if np.isclose(np.sum(h_opt1), target_sum, atol=1e-10) and \
               np.all(h_opt1 >= -1e-12) and np.all(h_opt1 <= 1 + 1e-12):
                current_score1 = score_h(h_opt1)
                if current_score1 < best_score:
                    best_score = current_score1
                    best_h = h_opt1.copy()
            
            if time.time() - start_time < time_limit - 5:
                h_opt2 = optimize_epigraph(best_h, beta, 50000.0, 800)
                if np.isclose(np.sum(h_opt2), target_sum, atol=1e-10) and \
                   np.all(h_opt2 >= -1e-12) and np.all(h_opt2 <= 1 + 1e-12):
                    current_score2 = score_h(h_opt2)
                    if current_score2 < best_score:
                        best_score = current_score2
                        best_h = h_opt2.copy()
    
    # Final constraint check and exact projection
    best_h = np.clip(best_h, 0, 1)
    for _ in range(100):
        diff = target_sum - np.sum(best_h)
        if abs(diff) < 1e-15:
            break
        if diff > 0:
            can_inc = best_h < 1.0
            if np.sum(can_inc) == 0:
                break
            # Distribute proportionally to headroom
            headroom = 1.0 - best_h[can_inc]
            if np.sum(headroom) > 0:
                best_h[can_inc] += diff * headroom / np.sum(headroom)
            else:
                best_h[can_inc] += diff / np.sum(can_inc)
        else:
            can_dec = best_h > 0.0
            if np.sum(can_dec) == 0:
                break
            # Distribute proportionally to current values
            weights = best_h[can_dec]
            if np.sum(weights) > 0:
                best_h[can_dec] += diff * weights / np.sum(weights)
            else:
                best_h[can_dec] += diff / np.sum(can_dec)
        best_h = np.clip(best_h, 0, 1)
    
    # Final verification: ensure sum is exactly target_sum
    final_diff = target_sum - np.sum(best_h)
    if abs(final_diff) > 1e-14:
        # Last resort: adjust the element closest to 0.5
        mid_idx = np.argmin(np.abs(best_h - 0.5))
        best_h[mid_idx] += final_diff
        best_h = np.clip(best_h, 0, 1)
        
    return best_h
