import numpy as np
import time
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def _conv_max_and_grad(h):
    """Compute max of autoconvolution and its gradient w.r.t. h (h already normalized to sum 1)."""
    conv = fftconvolve(h, h, mode='full')
    max_c = np.max(conv)
    # Gradient of conv[k] w.r.t. h[j]: d/dh[j] sum_i h[i]h[k-i] = h[k-j] + h[j-k] (when in bounds)
    # For the max element, we need the gradient of that particular conv element.
    # Actually for smooth max we use log-sum-exp. Let's return conv and max_c.
    return conv, max_c

def _smooth_max_conv(h, beta):
    """Smooth approximation of max(conv) using log-sum-exp."""
    conv = fftconvolve(h, h, mode='full')
    max_c = np.max(conv)
    shifted = conv - max_c
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50.0, 0.0)
    log_sum_exp = max_c + np.log(np.sum(np.exp(beta * shifted)))
    return log_sum_exp / beta

def _smooth_max_conv_grad(h, beta):
    """Gradient of smooth max of autoconvolution w.r.t. h."""
    conv = fftconvolve(h, h, mode='full')
    max_c = np.max(conv)
    shifted = conv - max_c
    shifted = np.clip(shifted, -50.0, 0.0)
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp  # softmax weights over convolution lags
    
    # d/dh[j] (sum_lag w_lag * conv[lag]) = sum_lag w_lag * (h[lag-j] + h[j-lag])
    # where indices are in [0, 2N-2] for conv, and h is [0, N-1]
    grad = np.zeros(N)
    for lag in range(len(conv)):
        wl = weights[lag]
        if wl < 1e-12:
            continue
        # h[lag - j] for j in [0, N-1]: valid when 0 <= lag-j < N
        # h[j - lag] for j in [0, N-1]: valid when 0 <= j-lag < N
        # Vectorized:
        # For term h[lag - j]: j ranges over max(0, lag-(N-1)) to min(N-1, lag)
        lo1 = max(0, lag - (N - 1))
        hi1 = min(N - 1, lag)
        if lo1 <= hi1:
            js = np.arange(lo1, hi1 + 1)
            idx = lag - js  # indices into h
            grad[js] += wl * h[idx]
        # For term h[j - lag]: j ranges over max(0, lag) to min(N-1, lag + N - 1)
        lo2 = max(0, lag)
        hi2 = min(N - 1, lag + N - 1)
        if lo2 <= hi2:
            js = np.arange(lo2, hi2 + 1)
            idx = js - lag  # indices into h
            grad[js] += wl * h[idx]
    
    # Multiply by 2 because conv = h*h (both factors contribute)
    # Wait, I already accounted for both terms above (h[lag-j] and h[j-lag])
    # So grad is already the full gradient. No extra factor of 2.
    return grad

def _project_to_simplex(v):
    """Project v onto the probability simplex {x >= 0, sum(x) = 1}."""
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, n + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        return np.ones(n) / n
    rho = rho[-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = np.sum(w)
    if s < 1e-15:
        return np.ones(n) / n
    return w / s

def _optimize_full_vector(beta, max_iter, lr_init, h_init=None):
    """
    Optimize h (50-dim) to minimize smooth max conv, projected to simplex.
    Returns (h, score).
    """
    if h_init is not None:
        h = np.maximum(h_init, 0.0)
        s = np.sum(h)
        if s < 1e-15:
            h = np.ones(N) / N
        else:
            h = h / s
    else:
        h = np.ones(N) / N
    
    best_h = h.copy()
    best_score = compute_ratio(h)
    
    lr = lr_init
    for it in range(max_iter):
        # Compute gradient of smooth max conv
        grad = _smooth_max_conv_grad(h, beta)
        
        # We want to minimize max conv (to increase ratio, since h2_sq is fixed-ish)
        # But ratio = h2_sq / max_conv. To maximize ratio, minimize max_conv / h2_sq.
        # Actually, let's directly maximize ratio. The gradient of ratio is complex.
        # Simpler: minimize smooth_max_conv, keeping h on simplex.
        # But this doesn't account for h2_sq. Let's use a different objective.
        
        # Alternative: directly maximize log(ratio) = log(h2_sq) - log(max_conv)
        # d/dh log(h2_sq) = 2h / h2_sq
        # d/dh log(max_conv) ≈ grad_max_conv / max_conv (using smooth approx)
        
        h2_sq = np.sum(h**2)
        smooth_max = _smooth_max_conv(h, beta)
        
        # Gradient of log(h2_sq) - log(smooth_max)
        if h2_sq < 1e-15 or smooth_max < 1e-15:
            break
        
        grad_h2 = 2.0 * h / h2_sq
        grad_max = grad / smooth_max  # gradient of log(smooth_max)
        
        # We want to maximize, so gradient ascent on log(ratio)
        grad_total = grad_h2 - grad_max
        
        # Projected gradient ascent on simplex
        h_new = h + lr * grad_total
        h_new = _project_to_simplex(h_new)
        
        sc_new = compute_ratio(h_new)
        if sc_new > best_score:
            best_h = h_new.copy()
            best_score = sc_new
            h = h_new
        else:
            lr *= 0.5
            if lr < 1e-8:
                break
    
    return best_h, best_score

def _local_polish(h, max_iter=200, tol=1e-10):
    """
    Local polish: coordinate descent + pair swaps to improve ratio.
    """
    h = np.maximum(h, 0.0)
    s = np.sum(h)
    if s < 1e-15:
        return h
    h = h / s
    
    best_score = compute_ratio(h)
    support = np.where(h > 1e-12)[0]
    
    for _ in range(max_iter):
        improved = False
        
        # Try adjusting pairs of support elements
        for i in range(len(support)):
            for j in range(i + 1, len(support)):
                # Try transferring mass from i to j and vice versa
                for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    # Move frac * h[i] from i to j
                    transfer = frac * h[support[i]]
                    h_try = h.copy()
                    h_try[support[i]] -= transfer
                    h_try[support[j]] += transfer
                    h_try = np.maximum(h_try, 0.0)
                    s_try = np.sum(h_try)
                    if s_try < 1e-15:
                        continue
                    h_try = h_try / s_try
                    sc_try = compute_ratio(h_try)
                    if sc_try > best_score + tol:
                        h = h_try
                        best_score = sc_try
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        # Also try adding mass to zero elements
        if not improved:
            for k in range(N):
                if h[k] < 1e-12:
                    # Try adding small mass to k, reducing from the largest element
                    max_idx = np.argmax(h)
                    for frac in [0.05, 0.1, 0.2]:
                        transfer = frac * h[max_idx]
                        h_try = h.copy()
                        h_try[max_idx] -= transfer
                        h_try[k] += transfer
                        h_try = np.maximum(h_try, 0.0)
                        s_try = np.sum(h_try)
                        if s_try < 1e-15:
                            continue
                        h_try = h_try / s_try
                        sc_try = compute_ratio(h_try)
                        if sc_try > best_score + tol:
                            h = h_try
                            best_score = sc_try
                            improved = True
                            break
                    if improved:
                        break
        
        if not improved:
            break
    
    return h

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin (150s limit)
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    rng = np.random.RandomState(42)
    
    # Strategy 1: Multiple random restarts with full-vector optimization
    n_restarts = 0
    while time.time() < deadline - 20.0:
        n_restarts += 1
        
        # Generate random initial point
        # Mix of sparse and dense initializations
        if n_restarts % 3 == 0:
            # Sparse: random subset of 5-15 elements
            k = rng.randint(5, 16)
            idx = rng.choice(N, k, replace=False)
            h_init = np.zeros(N)
            h_init[idx] = rng.dirichlet(np.ones(k))
        elif n_restarts % 3 == 1:
            # Dense: random Dirichlet
            h_init = rng.dirichlet(np.ones(N) * 0.5)
        else:
            # Sidon-like: use a known good structure
            # Spread out indices
            k = rng.randint(8, 14)
            idx = sorted(rng.choice(N, k, replace=False).tolist())
            h_init = np.zeros(N)
            h_init[idx] = rng.dirichlet(np.ones(k) * 2.0)
        
        # Try a few betas
        for beta in [30.0, 60.0, 120.0]:
            if time.time() > deadline - 20.0:
                break
            
            h_opt, sc = _optimize_full_vector(beta, max_iter=150, lr_init=0.05, h_init=h_init)
            
            # Polish
            h_polished = _local_polish(h_opt, max_iter=100, tol=1e-10)
            sc_polished = compute_ratio(h_polished)
            
            if sc_polished > best_score:
                best_score = sc_polished
                best_h = h_polished.copy()
        
        # Also try starting from current best with perturbation
        if n_restarts > 2 and time.time() < deadline - 20.0:
            h_perturbed = best_h + rng.randn(N) * 0.05
            h_perturbed = _project_to_simplex(h_perturbed)
            for beta in [60.0]:
                if time.time() > deadline - 20.0:
                    break
                h_opt, sc = _optimize_full_vector(beta, max_iter=100, lr_init=0.03, h_init=h_perturbed)
                h_polished = _local_polish(h_opt, max_iter=80, tol=1e-10)
                sc_polished = compute_ratio(h_polished)
                if sc_polished > best_score:
                    best_score = sc_polished
                    best_h = h_polished.copy()
    
    # Final polish
    if time.time() < deadline - 5.0:
        best_h = _local_polish(best_h, max_iter=200, tol=1e-11)
        final_score = compute_ratio(best_h)
        if final_score > best_score:
            best_score = final_score
    
    return list(best_h)
