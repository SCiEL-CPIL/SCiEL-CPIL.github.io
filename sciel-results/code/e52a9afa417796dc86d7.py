import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _overlaps_fft(h):
    """Compute all cyclic overlaps of h with shifted (1-h) using FFT."""
    one_minus_h = 1.0 - h
    # overlap[k] = sum_i h[i] * (1-h[(i+k) % N])
    # This is the circular cross-correlation of h and (1-h)
    H = np.fft.rfft(h)
    G = np.fft.rfft(one_minus_h)
    # circular correlation: ifft(conj(H) * G) gives sum_i h[i] * g[(i+k)%N]
    corr = np.fft.irfft(np.conj(H) * G, n=N)
    return corr

def _soft_max_overlaps(corr, alpha):
    """Soft maximum of corr using log-sum-exp."""
    # Shift for numerical stability
    m = np.max(corr)
    if m < -1e15:
        return 0.0
    shifted = corr - m
    # soft max = m + (1/alpha) * log(sum(exp(alpha * shifted)))
    exp_shifted = np.exp(alpha * shifted)
    s = np.sum(exp_shifted)
    if s < 1e-300:
        return m
    return m + np.log(s) / alpha

def _soft_max_overlaps_grad(corr, alpha):
    """Return (soft_max_value, gradient w.r.t. corr)."""
    m = np.max(corr)
    shifted = corr - m
    exp_shifted = np.exp(alpha * shifted)
    s = np.sum(exp_shifted)
    if s < 1e-300:
        val = m
        grad = np.zeros_like(corr)
        idx = np.argmax(corr)
        grad[idx] = 1.0
    else:
        val = m + np.log(s) / alpha
        grad = exp_shifted / s
    return val, grad

def _objective_and_grad(h, alpha):
    """Compute soft-max overlap objective and its gradient w.r.t. h."""
    one_minus_h = 1.0 - h
    corr = _overlaps_fft(h)
    val, corr_grad = _soft_max_overlaps_grad(corr, alpha)
    
    # d corr[k] / d h[i] = (1 - h[(i+k)%N]) for the term where index matches
    # corr[k] = sum_i h[i] * (1 - h[(i+k)%N])
    # d corr[k] / d h[j] = (1 - h[(j+k)%N]) + h[j] * (-1) * delta_{(i+k)%N, j}... 
    # Actually: corr[k] = sum_i h[i] * (1-h[(i+k)%N])
    # d corr[k] / d h[j] = (1 - h[(j+k)%N]) + sum_i h[i] * d(1-h[(i+k)%N])/dh[j]
    # The second term: d(1-h[(i+k)%N])/dh[j] = -1 if (i+k)%N == j, i.e., i = (j-k)%N
    # So second term = -h[(j-k)%N]
    # d corr[k] / d h[j] = (1 - h[(j+k)%N]) - h[(j-k)%N]
    
    # grad_h[j] = sum_k corr_grad[k] * d corr[k] / d h[j]
    # = sum_k corr_grad[k] * ((1 - h[(j+k)%N]) - h[(j-k)%N])
    
    # Compute efficiently:
    # term1[j] = sum_k corr_grad[k] * (1 - h[(j+k)%N])
    #   = sum_k corr_grad[k] - sum_k corr_grad[k] * h[(j+k)%N]
    #   = sum(corr_grad) - (corr_grad circularly correlated with h)[j]... 
    #   Actually term1[j] = sum_k corr_grad[k] * (1 - h[(j+k)%N])
    #   Let's define: for fixed j, sum_k corr_grad[k] * h[(j+k)%N]
    #   This is the circular cross-correlation of corr_grad and h evaluated at j
    #   = ifft(conj(FFT(corr_grad)) * FFT(h))[j]... 
    #   Wait: cross_corr(a, b)[j] = sum_k a[k] * b[(j+k)%N]
    #   = ifft(conj(FFT(a)) * FFT(b))[j]
    
    CG = np.fft.rfft(corr_grad)
    H = np.fft.rfft(h)
    
    # term1[j] = sum_k corr_grad[k] * (1 - h[(j+k)%N])
    # = sum_k corr_grad[k] - sum_k corr_grad[k] * h[(j+k)%N]
    # = sum(corr_grad) - cross_corr(corr_grad, h)[j]
    cross1 = np.fft.irfft(np.conj(CG) * H, n=N)
    term1 = np.sum(corr_grad) - cross1
    
    # term2[j] = sum_k corr_grad[k] * h[(j-k)%N]
    # = cross_corr_rev: sum_k corr_grad[k] * h[(j-k)%N]
    # = ifft(FFT(corr_grad) * conj(FFT(h)))[j]... let me think
    # sum_k a[k] * b[(j-k)%N] = ifft(FFT(a) * conj(FFT(b)))[j]
    cross2 = np.fft.irfft(CG * np.conj(H), n=N)
    term2 = cross2
    
    grad_h = term1 - term2
    
    return val, grad_h

def _project(h):
    """Project h onto {x : sum(x) = N/2, 0 <= x <= 1}."""
    h = np.clip(h, 0, 1)
    s = h.sum()
    if s > 0:
        h = h * (N / 2.0) / s
    return np.clip(h, 0, 1)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s
    
    def time_left():
        return deadline - time.time()
    
    # Initial candidates
    candidates = []
    
    # Uniform
    candidates.append(np.full(N, 0.5))
    
    # Alternating
    h_alt = np.zeros(N)
    h_alt[1::2] = 1.0
    candidates.append(h_alt.copy())
    
    # Random
    rng = np.random.default_rng(42)
    for seed in range(5):
        r = np.random.default_rng(seed)
        h_r = r.random(N)
        h_r = h_r * (N / 2.0) / h_r.sum()
        h_r = np.clip(h_r, 0, 1)
        candidates.append(h_r)
    
    # Evaluate initial candidates
    best_h = None
    best_score = float('inf')
    for h in candidates:
        h = _project(h)
        corr = _overlaps_fft(h)
        s = np.max(corr)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Projected gradient descent with adaptive alpha for soft-max
    # Start with large alpha (sharper max) and gradually increase
    alpha_start = 5.0
    alpha_end = 50.0
    total_iters = 3000
    
    for trial in range(3):
        if time_left() < 10:
            break
        
        # Pick starting point
        if trial == 0:
            h_cur = best_h.copy()
        else:
            r = rng.random(N)
            h_cur = _project(r)
        
        # Adaptive projected gradient descent
        alpha = alpha_start
        lr = 0.1
        best_trial_score = float('inf')
        best_trial_h = h_cur.copy()
        
        for it in range(total_iters):
            if time_left() < 5:
                break
            
            # Animate alpha: start smooth, end sharp
            progress = it / total_iters
            alpha = alpha_start + (alpha_end - alpha_start) * progress**2
            
            obj, grad = _objective_and_grad(h_cur, alpha)
            
            # Also compute true max overlap for tracking
            corr = _overlaps_fft(h_cur)
            true_max = np.max(corr)
            if true_max < best_trial_score:
                best_trial_score = true_max
                best_trial_h = h_cur.copy()
            
            # Adaptive step size: use gradient norm
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            
            # Step with backtracking
            for _ in range(20):
                h_new = h_cur - lr * grad
                h_new = _project(h_new)
                obj_new, _ = _objective_and_grad(h_new, alpha)
                if obj_new < obj + 1e-12:
                    break
                lr *= 0.5
            
            h_cur = h_new
        
        if best_trial_score < best_score:
            best_score = best_trial_score
            best_h = best_trial_h.copy()
    
    # Final refinement with very sharp alpha
    if time_left() > 10:
        alpha_final = 100.0
        h_cur = best_h.copy()
        for it in range(500):
            if time_left() < 3:
                break
            obj, grad = _objective_and_grad(h_cur, alpha_final)
            corr = _overlaps_fft(h_cur)
            true_max = np.max(corr)
            if true_max < best_score:
                best_score = true_max
                best_h = h_cur.copy()
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            
            lr = 0.05
            for _ in range(10):
                h_new = h_cur - lr * grad
                h_new = _project(h_new)
                obj_new, _ = _objective_and_grad(h_new, alpha_final)
                if obj_new < obj + 1e-12:
                    break
                lr *= 0.5
            h_cur = h_new
    
    # Final projection
    best_h = _project(best_h)
    
    return best_h
