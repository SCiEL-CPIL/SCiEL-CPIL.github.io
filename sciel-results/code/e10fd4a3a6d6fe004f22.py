"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def _autoconv(h):
    """Compute autoconvolution of h on the grid [-1/4, 1/4]."""
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n_fft)
    conv = np.fft.irfft(H * H, n_fft)[:2 * N - 1]
    return conv

def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0) or np.sum(h) < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    conv = _autoconv(h)
    sup = np.max(conv)
    if sup < 1e-30:
        return 0.0
    return l2_sq / (l1 * sup)

def _soft_max(x, beta=50.0):
    """Soft maximum via log-sum-exp."""
    m = np.max(x)
    if m < 1e-30:
        return 1e-30
    return m + np.log(np.sum(np.exp((x - m) * beta)) / len(x)) / beta

def _log_ratio_and_grad(h, beta=50.0):
    """Compute log(ratio) and its gradient with respect to h.
    
    ratio = l2_sq / (l1 * sup)
    log(ratio) = log(l2_sq) - log(l1) - log(sup)
    
    For soft sup, we use:
    soft_sup = max + (1/beta)*log(sum(exp(beta*(x-max))))
    d(soft_sup)/dx_i = (1/soft_sup) * exp(beta*(x_i-soft_sup)) * (soft_sup - max)
    Actually simpler: d(soft_sup)/dx_i = exp(beta*(x_i - soft_sup)) / sum_j exp(beta*(x_j - soft_sup))
    """
    h = np.asarray(h, dtype=float)
    n = len(h)
    
    # l2_sq = sum(h_i^2), d(l2_sq)/dh_i = 2*h_i
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    
    if l1 < 1e-15 or l2_sq < 1e-30:
        return -1e30, np.zeros(n)
    
    conv = _autoconv(h)
    soft_sup = _soft_max(conv, beta)
    
    if soft_sup < 1e-30:
        return -1e30, np.zeros(n)
    
    log_r = np.log(l2_sq) - np.log(l1) - np.log(soft_sup)
    
    # Gradient computation
    # d(log_r)/dh_i = (1/l2_sq)*d(l2_sq)/dh_i - (1/l1)*d(l1)/dh_i - (1/soft_sup)*d(soft_sup)/dh_i
    # = 2*h_i/l2_sq - 1/l1 - (1/soft_sup) * d(soft_sup)/dh_i
    
    # d(soft_sup)/dh_i: soft_sup is a function of conv, which is a function of h
    # conv_k = sum_{i+j=k} h_i * h_j (with proper indexing)
    # d(conv_k)/dh_i = sum_j h_j where i+j=k, plus sum_j h_j where j+i=k (same thing for symmetric)
    # Actually for autoconv: conv_k = sum_i h_i * h_{k-i}
    # d(conv_k)/dh_i = h_{k-i} + h_{k-i} (if i != k-i) or 2*h_i (if i = k-i)
    # More carefully: d(conv_k)/dh_i = sum_j [delta_{i,j} * h_{k-j} + h_j * delta_{i,k-j}]
    # = h_{k-i} + h_{i} (when k-i = i, i.e., k=2i)
    # Actually: conv_k = sum_i h_i * h_{k-i}
    # d(conv_k)/dh_i = h_{k-i} + (1 if k-i == i else 0) * h_i
    # Wait, that's not right. Let me think again.
    # conv_k = sum_{i=0}^{k} h_i * h_{k-i} (for valid indices)
    # d(conv_k)/dh_i = h_{k-i} + h_{k-i} (if i appears twice, i.e., 2i=k)
    # No: d/dh_i of sum_j h_j * h_{k-j} = h_{k-i} + (1 if k-i=i else 0)*h_i
    # Actually just: d/dh_i of h_i * h_{k-i} = h_{k-i}
    # And d/dh_i of h_{k-i} * h_i = h_{k-i}
    # So if i != k-i, both terms contribute h_{k-i}
    # If i = k-i, then it's h_i^2, derivative is 2*h_i
    
    # For soft_sup, d(soft_sup)/dh_i = sum_k w_k * d(conv_k)/dh_i
    # where w_k = exp(beta*(conv_k - soft_sup)) / sum_j exp(beta*(conv_j - soft_sup))
    
    # Compute weights
    shifted = beta * (conv - soft_sup)
    # Clip for numerical stability
    shifted = np.clip(shifted, -100, 100)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # Now compute d(soft_sup)/dh_i = sum_k weights[k] * d(conv_k)/dh_i
    # conv_k = sum_i h_i * h_{k-i} for valid i, k-i
    # d(conv_k)/dh_i = h_{k-i} if k-i is in [0, N-1] and i is in [0, N-1]
    # plus another h_{k-i} if i == k-i (i.e., k = 2i)
    
    # Let's compute this efficiently
    d_soft_sup_dh = np.zeros(n)
    for k in range(len(conv)):
        if weights[k] < 1e-15:
            continue
        w_k = weights[k]
        # For this k, conv_k involves pairs (i, k-i) where 0<=i<N and 0<=k-i<N
        i_min = max(0, k - (N - 1))
        i_max = min(N - 1, k)
        for i in range(i_min, i_max + 1):
            j = k - i
            if j < 0 or j >= N:
                continue
            if i == j:
                d_soft_sup_dh[i] += w_k * 2 * h[i]
            else:
                d_soft_sup_dh[i] += w_k * h[j]
    
    grad = 2 * h / l2_sq - 1.0 / l1 - d_soft_sup_dh / soft_sup
    
    return log_r, grad

def _local_optimize_gradient(h0, max_iters=500, beta=50.0, lr=0.1):
    """Gradient-based local optimization with adaptive learning rate."""
    h = np.array(h0, dtype=float)
    h = np.maximum(h, 0)
    h = h / (np.sum(h) + 1e-30)  # normalize
    
    best_h = h.copy()
    best_log_r, _ = _log_ratio_and_grad(h, beta)
    best_r = _ratio(h)
    
    lr_current = lr
    stuck_count = 0
    
    for it in range(max_iters):
        if time.time() > _deadline - 5:
            break
        
        log_r, grad = _log_ratio_and_grad(h, beta)
        
        # Gradient ascent on log_r (we want to maximize)
        # Apply gradient with adaptive step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        # Scale gradient
        step = grad / grad_norm * lr_current
        
        # Update
        h_new = h + step
        h_new = np.maximum(h_new, 0)
        
        # Check improvement
        new_log_r, _ = _log_ratio_and_grad(h_new, beta)
        
        if new_log_r > log_r + 1e-10:
            h = h_new
            lr_current = min(lr_current * 1.1, 1.0)
            stuck_count = 0
        else:
            lr_current = max(lr_current * 0.5, 0.001)
            stuck_count += 1
            if stuck_count > 50:
                break
        
        # Track best
        r_check = _ratio(h)
        if r_check > best_r:
            best_r = r_check
            best_h = h.copy()
    
    return best_h, best_r

def _local_optimize(h0, max_iters=200, tol=1e-8):
    """Combined local optimization: gradient-based then Nelder-Mead polish."""
    h = np.array(h0, dtype=float)
    h = np.maximum(h, 0)
    
    best_h = h.copy()
    best_r = _ratio(h)
    
    # Phase 1: Gradient-based optimization
    if time.time() < _deadline - 20:
        h_grad, r_grad = _local_optimize_gradient(h, max_iters=min(max_iters, 300))
        if r_grad > best_r:
            best_h = h_grad
            best_r = r_grad
    
    # Phase 2: Nelder-Mead polish
    if time.time() < _deadline - 10:
        try:
            from scipy.optimize import minimize
            
            def neg_ratio(h):
                h = np.maximum(h, 0)
                return -_ratio(h)
            
            res = minimize(neg_ratio, best_h, method='Nelder-Mead', 
                          options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10})
            h_opt = np.maximum(res.x, 0)
            r_opt = _ratio(h_opt)
            if r_opt > best_r:
                best_h = h_opt
                best_r = r_opt
        except:
            pass
    
    return best_h, best_r

def _gaussian_profile(center, width, amplitude, N, grid_start, grid_end):
    """Generate a Gaussian profile on the grid."""
    x = np.linspace(grid_start, grid_end, N)
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)

def _lorentzian_profile(center, width, amplitude, N, grid_start, grid_end):
    """Generate a Lorentzian (Cauchy) profile on the grid."""
    x = np.linspace(grid_start, grid_end, N)
    return amplitude / (1 + ((x - center) / width) ** 2)

_deadline = 0

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 140  # safety margin
    
    grid_start = -0.25
    grid_end = 0.25
    
    best_h = np.ones(N)
    best_r = _ratio(best_h)
    
    # Phase 1: Structured parametric search
    centers = np.linspace(-0.2, 0.2, 9)
    widths = np.linspace(0.02, 0.25, 12)
    amplitudes = [1.0]
    
    for profile_func in [_gaussian_profile, _lorentzian_profile]:
        for c in centers:
            for w in widths:
                if time.time() > _deadline:
                    break
                for amp in amplitudes:
                    h = profile_func(c, w, amp, N, grid_start, grid_end)
                    h = np.maximum(h, 0)
                    if np.sum(h) < 1e-10:
                        continue
                    r = _ratio(h)
                    if r > best_r:
                        best_r = r
                        best_h = h.copy()
        
        if time.time() > _deadline:
            break
    
    # Phase 2: Local optimization from best structured result
    if time.time() < _deadline - 15:
        h_opt, r_opt = _local_optimize(best_h, max_iters=400)
        if r_opt > best_r:
            best_h = h_opt
            best_r = r_opt
    
    # Phase 3: Try additional local optimizations from diverse starting points
    if time.time() < _deadline - 10:
        diverse_starts = []
        # Uniform
        diverse_starts.append(np.ones(N))
        # Triangular
        diverse_starts.append(np.abs(np.linspace(-1, 1, N)))
        # Two peaks
        h2 = np.zeros(N)
        h2[:N//4] = 1.0
        h2[3*N//4:] = 1.0
        diverse_starts.append(h2)
        # Narrow center peak
        h3 = np.zeros(N)
        mid = N // 2
        for i in range(N):
            h3[i] = np.exp(-0.5 * ((i - mid) / 3.0) ** 2)
        diverse_starts.append(h3)
        # Broad center peak
        h4 = np.zeros(N)
        for i in range(N):
            h4[i] = np.exp(-0.5 * ((i - mid) / 8.0) ** 2)
        diverse_starts.append(h4)
        
        for h0 in diverse_starts:
            if time.time() > _deadline - 10:
                break
            h_opt, r_opt = _local_optimize(h0, max_iters=200)
            if r_opt > best_r:
                best_h = h_opt
                best_r = r_opt
    
    # Phase 4: Final high-precision polish
    if time.time() < _deadline - 5:
        h_opt, r_opt = _local_optimize(best_h, max_iters=300)
        if r_opt > best_r:
            best_h = h_opt
            best_r = r_opt
    
    # Ensure final result is valid
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return list(best_h)
# EVOLVE-BLOCK-END
