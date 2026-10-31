import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 140.0  # 150s limit, 10s margin

def compute_ratio(h):
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-9):
        return -1.0
    h = np.clip(h, 0.0, None)
    l1 = h.sum()
    if l1 < 1e-12:
        return -1.0
    l2_sq = float(np.dot(h, h))
    conv = np.convolve(h, h, mode='full')
    max_conv = float(np.max(conv))
    if max_conv < 1e-12:
        return -1.0
    return l2_sq / (l1 * max_conv)

def is_sidon(arr):
    arr = sorted(set(arr))
    n = len(arr)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def find_sidon_greedy(max_val):
    sidon = []
    for i in range(max_val):
        if is_sidon(sidon + [i]):
            sidon.append(i)
    return sidon

def _soft_max_surrogate(params, beta):
    """
    Minimize -log(ratio) = log(l1 * max_conv) - log(l2_sq)
    using log-sum-exp for max_conv.
    """
    h = np.exp(params)
    l1 = h.sum()
    if l1 < 1e-30:
        return 1e6
    h = h / l1  # normalize to l1=1 (scale invariant)
    l2_sq = float(np.dot(h, h))
    if l2_sq < 1e-30:
        return 1e6
    conv = np.convolve(h, h, mode='full')
    # log-sum-exp for max
    m = np.max(conv)
    if m < 1e-30:
        return 1e6
    log_max = m + np.log(np.sum(np.exp(conv - m)))
    # -log(ratio) = log(l1 * max_conv) - log(l2_sq) = log(max_conv) - log(l2_sq) since l1=1
    return log_max - np.log(l2_sq)

def _exact_neg_log_ratio(params):
    """Exact (nonsmooth) objective for polish."""
    h = np.exp(params)
    l1 = h.sum()
    if l1 < 1e-30:
        return 1e6
    h = h / l1
    l2_sq = float(np.dot(h, h))
    if l2_sq < 1e-30:
        return 1e6
    conv = np.convolve(h, h, mode='full')
    max_conv = float(np.max(conv))
    if max_conv < 1e-30:
        return 1e6
    return np.log(max_conv) - np.log(l2_sq)

def construct():
    t0 = time.time()
    
    # --- Initial candidates ---
    candidates = []
    
    # Uniform
    h_uni = np.ones(N) / N
    candidates.append(h_uni)
    
    # Gaussian-like
    x = np.linspace(-1, 1, N)
    for sigma in [0.3, 0.5, 0.7, 1.0]:
        h_g = np.exp(-x**2 / (2 * sigma**2))
        h_g /= h_g.sum()
        candidates.append(h_g)
    
    # Cosine bump
    h_cos = 0.5 * (1 + np.cos(np.pi * x))
    h_cos /= h_cos.sum()
    candidates.append(h_cos)
    
    # Sidon-based sparse
    sidon_idx = find_sidon_greedy(N)
    if sidon_idx:
        h_sidon = np.zeros(N)
        h_sidon[sidon_idx] = 1.0 / np.sqrt(len(sidon_idx))
        candidates.append(h_sidon)
        
    # Delta function (single spike) - Known strong baseline
    for i in range(N):
        h_delta = np.zeros(N)
        h_delta[i] = 1.0
        candidates.append(h_delta)
        
    # Near-delta (two adjacent spikes)
    for i in range(N - 1):
        h_near_delta = np.zeros(N)
        h_near_delta[i] = 0.5
        h_near_delta[i+1] = 0.5
        candidates.append(h_near_delta)

    # Sparse random
    np.random.seed(42)
    for K in [4, 6, 8, 10, 12, 15]:
        idx = np.random.choice(N, K, replace=False)
        h_sp = np.zeros(N)
        h_sp[idx] = 1.0 / np.sqrt(K)
        candidates.append(h_sp)
    
    # Evaluate candidates
    best_h = candidates[0].copy()
    best_val = compute_ratio(best_h)
    for h in candidates[1:]:
        v = compute_ratio(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
    
    # --- Phase 1: Smooth optimization (log-sum-exp surrogate) ---
    def smooth_obj(params):
        return _soft_max_surrogate(params, beta=50.0)
    
    n_restarts = 30
    for r in range(n_restarts):
        if time.time() > DEADLINE - 20:
            break
        
        if r == 0:
            x0 = np.log(best_h + 1e-8)
        elif r == 1 and sidon_idx:
            h_s = np.zeros(N)
            h_s[sidon_idx] = 1.0 / np.sqrt(len(sidon_idx))
            x0 = np.log(h_s + 1e-8)
        else:
            # Random sparse start
            K = np.random.randint(3, 16)
            idx = np.random.choice(N, K, replace=False)
            h_start = np.zeros(N)
            h_start[idx] = 1.0 / np.sqrt(K)
            x0 = np.log(h_start + 1e-8)
        
        try:
            res = minimize(smooth_obj, x0, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < 1e5:
                h_opt = np.exp(res.x)
                h_opt = h_opt / h_opt.sum()
                v = compute_ratio(h_opt)
                if v > best_val:
                    best_val = v
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # --- Phase 2: High-precision polish with exact objective ---
    if time.time() < DEADLINE - 5:
        x0 = np.log(best_h + 1e-8)
        try:
            res = minimize(_exact_neg_log_ratio, x0, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < 1e5:
                h_opt = np.exp(res.x)
                h_opt = h_opt / h_opt.sum()
                v = compute_ratio(h_opt)
                if v > best_val:
                    best_val = v
                    best_h = h_opt.copy()
        except Exception:
            pass
            
    # --- Phase 3: Local Perturbation Search (Hill Climbing) ---
    # Replaces the slow coordinate descent with a more effective random walk
    if time.time() < DEADLINE - 2:
        h_curr = best_h.copy()
        v_curr = best_val
        np.random.seed(123)
        step_size = 0.1
        no_improve_count = 0
        
        while time.time() < DEADLINE - 2:
            # Generate a perturbed vector
            h_try = h_curr.copy()
            
            # Randomly select a few indices to perturb
            num_perturb = np.random.randint(1, 4)
            indices = np.random.choice(N, num_perturb, replace=False)
            
            for idx in indices:
                # Perturb value: add random noise
                # Keep non-negative
                noise = np.random.normal(0, step_size * (1.0 + h_curr[idx]))
                h_try[idx] = max(0.0, h_curr[idx] + noise)
                
            if h_try.sum() < 1e-12:
                continue
                
            h_try = h_try / h_try.sum()
            v_try = compute_ratio(h_try)
            
            if v_try > v_curr + 1e-10:
                h_curr = h_try.copy()
                v_curr = v_try
                best_val = v_try
                best_h = h_curr.copy()
                no_improve_count = 0
                # Slowly decrease step size
                step_size *= 0.99
            else:
                no_improve_count += 1
                # If stuck for a while, increase step size to escape local minimum
                if no_improve_count > 100:
                    step_size *= 1.2
                    no_improve_count = 0
                    
            # Cap step size
            if step_size > 0.5: step_size = 0.5
            if step_size < 1e-4: step_size = 1e-4

    # Final safety
    best_h = np.clip(best_h, 0.0, None)
    if best_h.sum() < 1e-12:
        best_h = np.ones(N) / N
    
    return list(best_h)
