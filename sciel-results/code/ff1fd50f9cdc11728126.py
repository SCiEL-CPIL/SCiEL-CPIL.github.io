import time
import numpy as np
from scipy.optimize import minimize, linprog, minimize_scalar

N = 600
DEADLINE = 110.0  # Safety margin before 120s kill

def compute_score(heights):
    """Compute the normalized maximum of autoconvolution for a step function.
    Returns the value such that uniform heights give score 1.0.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    
    n = len(heights)
    w = 0.5 / n  # width of each interval
    
    # Use FFT for linear autoconvolution
    M = 1
    while M < 2 * n - 1:
        M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
    
    score = max_conv * n / (total * total * w)
    return score

def _dct_matrix(n, k):
    """Create DCT-II matrix for n points, first k+1 coefficients."""
    idx = np.arange(n)
    DCT = np.zeros((n, k + 1))
    for m in range(k + 1):
        DCT[:, m] = np.cos(np.pi * m * (2 * idx + 1) / (2 * n))
    return DCT

def _idct_coefficients_to_heights(c, n, k):
    """Reconstruct heights from DCT coefficients (c[0..k])."""
    idx = np.arange(n)
    heights = np.zeros(n)
    for m in range(k + 1):
        heights += c[m] * np.cos(np.pi * m * (2 * idx + 1) / (2 * n))
    return heights

def _autoconv_and_grad(heights, M=None):
    """Compute autoconvolution and gradient of max autoconvolution w.r.t. heights."""
    if M is None:
        n = len(heights)
        M = 1
        while M < 2 * n - 1:
            M *= 2
    
    f = np.zeros(M, dtype=np.float64)
    f[:len(heights)] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=M)
    
    peak_idx = np.argmax(conv)
    
    n = len(heights)
    grad = np.zeros(n, dtype=np.float64)
    
    i_min = max(0, peak_idx - n + 1)
    i_max = min(n - 1, peak_idx)
    if i_min <= i_max:
        indices = np.arange(i_min, i_max + 1)
        j_indices = peak_idx - indices
        grad[indices] = 2.0 * heights[j_indices]
    
    return conv, grad, peak_idx

def _dct_optimize(start_time, deadline):
    """Optimize DCT coefficients to minimize peak autoconvolution."""
    def time_left():
        return deadline - time.time()
    
    w = 0.5 / N
    M = 1
    while M < 2 * N - 1:
        M *= 2
    
    def compute_score_fast(heights):
        f = np.zeros(M, dtype=np.float64)
        f[:N] = heights
        F = np.fft.rfft(f)
        conv = np.fft.irfft(F * F, n=M)
        max_conv = np.max(conv)
        total = np.sum(heights)
        if total <= 0:
            return np.inf
        return max_conv * N / (total * total * w)
    
    best_score = np.inf
    best_h = None
    
    # Try different numbers of DCT coefficients
    for k in [5, 10, 20, 30]:
        if time_left() < 20:
            break
        
        # Initialize with uniform signal (only DC component)
        c_init = np.zeros(k + 1)
        c_init[0] = np.sqrt(2 / N)  # Normalize
        
        def dct_objective(c):
            heights = _idct_coefficients_to_heights(c, N, k)
            # Penalty for negative values
            neg_penalty = np.sum(np.minimum(heights, 0) ** 2) * 100
            if np.sum(heights) <= 0:
                return 1e10
            score = compute_score_fast(np.maximum(heights, 0))
            return score + neg_penalty
        
        def dct_grad(c):
            heights = _idct_coefficients_to_heights(c, N, k)
            heights_clipped = np.maximum(heights, 0)
            
            if np.sum(heights_clipped) <= 0:
                return np.zeros(k + 1)
            
            # Get gradient of score w.r.t. heights
            f = np.zeros(M, dtype=np.float64)
            f[:N] = heights_clipped
            F = np.fft.rfft(f)
            conv = np.fft.irfft(F * F, n=M)
            peak_idx = np.argmax(conv)
            
            grad_h = np.zeros(N)
            i_min = max(0, peak_idx - N + 1)
            i_max = min(N - 1, peak_idx)
            if i_min <= i_max:
                indices = np.arange(i_min, i_max + 1)
                j_indices = peak_idx - indices
                grad_h[indices] = 2.0 * heights_clipped[j_indices]
            
            S = np.sum(heights_clipped)
            obj_grad_h = grad_h * (N / (S * S * w))
            
            # Gradient w.r.t. DCT coefficients
            idx = np.arange(N)
            grad_c = np.zeros(k + 1)
            for m in range(k + 1):
                grad_c[m] = np.sum(grad_h * np.cos(np.pi * m * (2 * idx + 1) / (2 * N)))
            
            # Add penalty gradient
            neg_mask = heights < 0
            if np.any(neg_mask):
                for m in range(k + 1):
                    grad_c[m] += 2 * 100 * np.sum(np.minimum(heights, 0) * np.cos(np.pi * m * (2 * idx + 1) / (2 * N)))
            
            return grad_c
        
        # Use L-BFGS-B optimizer
        try:
            result = minimize(
                dct_objective,
                c_init,
                method='L-BFGS-B',
                jac=dct_grad,
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            c_opt = result.x
            h_opt = _idct_coefficients_to_heights(c_opt, N, k)
            h_opt = np.maximum(h_opt, 0.0)
            
            if np.sum(h_opt) > 0:
                score = compute_score_fast(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    return best_score, best_h

def _two_block_optimize(start_time, deadline):
    """Optimize two-block symmetric shape: h = [a]*m + [b]*(N-2m) + [a]*m."""
    def time_left():
        return deadline - time.time()
    
    w = 0.5 / N
    M = 1
    while M < 2 * N - 1:
        M *= 2
    
    def compute_score_fast(heights):
        f = np.zeros(M, dtype=np.float64)
        f[:N] = heights
        F = np.fft.rfft(f)
        conv = np.fft.irfft(F * F, n=M)
        max_conv = np.max(conv)
        total = np.sum(heights)
        if total <= 0:
            return np.inf
        return max_conv * N / (total * total * w)
    
    best_score = np.inf
    best_h = None
    
    # Try different block sizes
    for m in range(1, N // 4 + 1, 10):
        if time_left() < 10:
            break
        
        # Optimize over a and b
        def two_block_score(params):
            a, b = params
            if a < 0 or b < 0:
                return 1e10
            h = np.zeros(N)
            h[:m] = a
            h[m:N-m] = b
            h[N-m:] = a
            return compute_score_fast(h)
        
        # Grid search over a/b ratio
        for ratio in np.linspace(0.1, 2.0, 20):
            # Try different scales
            for scale in np.linspace(0.5, 2.0, 10):
                a = scale
                b = a / ratio
                if b < 0:
                    continue
                h = np.zeros(N)
                h[:m] = a
                h[m:N-m] = b
                h[N-m:] = a
                score = compute_score_fast(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
    
    return best_score, best_h

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def time_left():
        return deadline - time.time()
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Compute M once for reuse
    M = 1
    while M < 2 * N - 1:
        M *= 2
    
    # Phase 1: DCT-based optimization (new structural approach)
    if time_left() > 40:
        s_dct, h_dct = _dct_optimize(start_time, deadline)
        if h_dct is not None and s_dct < best_score:
            best_score = s_dct
            best_h = h_dct.copy()
    
    # Phase 2: Two-block optimization
    if time_left() > 30:
        s_2b, h_2b = _two_block_optimize(start_time, deadline)
        if h_2b is not None and s_2b < best_score:
            best_score = s_2b
            best_h = h_2b.copy()
    
    # Phase 3: Gradient-based optimization with smooth surrogate
    if time_left() > 20:
        def smooth_max(conv, alpha):
            m = np.max(conv)
            if m < 0:
                m = 0
            exp_conv = np.exp(alpha * (conv - m))
            return m + np.log(np.sum(exp_conv)) / alpha
        
        def smooth_obj_and_grad(heights, alpha):
            f = np.zeros(M, dtype=np.float64)
            f[:N] = heights
            F = np.fft.rfft(f)
            conv = np.fft.irfft(F * F, n=M)
            
            m = np.max(conv)
            if m < 0:
                m = 0
            exp_conv = np.exp(alpha * (conv - m))
            weights = exp_conv / np.sum(exp_conv)
            
            sm = smooth_max(conv, alpha)
            
            # Gradient of smooth_max w.r.t. f
            W_full = np.fft.rfft(weights)
            F_full = np.fft.rfft(f)
            corr = np.fft.irfft(W_full * np.conj(F_full), n=M)
            grad_f = 2.0 * corr[:N]
            
            S = np.sum(heights)
            w = 0.5 / N
            if S <= 1e-10:
                return 1e10, np.zeros(N)
            
            obj = sm / (S * w) ** 2 * (N * w)
            obj_grad = grad_f * (N / (S * S * w)) - sm * (2.0 * N / (S * S * S * w))
            
            return obj, obj_grad
        
        h_opt = best_h.copy()
        alpha = 50.0
        lr = 0.01
        
        for it in range(300):
            if time_left() < 10:
                break
            
            obj, grad = smooth_obj_and_grad(h_opt, alpha)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break
            
            h_opt = h_opt - lr * grad
            h_opt = np.maximum(h_opt, 0.0)
            
            score = compute_score(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
            
            if it % 20 == 0:
                alpha = min(alpha * 1.2, 500.0)
                lr = max(lr * 0.9, 0.001)
    
    # Final polish
    if time_left() > 10:
        h_final = best_h.copy()
        alpha_f = 200.0
        lr_f = 0.001
        
        for it in range(50):
            if time_left() < 5:
                break
            
            f = np.zeros(M, dtype=np.float64)
            f[:N] = h_final
            F = np.fft.rfft(f)
            conv = np.fft.irfft(F * F, n=M)
            
            m = np.max(conv)
            if m < 0:
                m = 0
            exp_conv = np.exp(alpha_f * (conv - m))
            weights = exp_conv / np.sum(exp_conv)
            sm = m + np.log(np.sum(exp_conv)) / alpha_f
            
            W_full = np.fft.rfft(weights)
            F_full = np.fft.rfft(f)
            corr = np.fft.irfft(W_full * np.conj(F_full), n=M)
            grad_f = 2.0 * corr[:N]
            
            S = np.sum(h_final)
            w = 0.5 / N
            if S <= 1e-10:
                break
            
            obj_grad = grad_f * (N / (S * S * w)) - sm * (2.0 * N / (S * S * S * w))
            
            h_final = h_final - lr_f * obj_grad
            h_final = np.maximum(h_final, 0.0)
            
            score = compute_score(h_final)
            if score < best_score:
                best_score = score
                best_h = h_final.copy()
            
            lr_f = max(lr_f * 0.98, 0.0001)
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
