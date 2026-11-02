import time
import numpy as np

N = 600
DEADLINE = 135.0  # Safety margin before 150s kill

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
    # Zero-pad to at least 2*n-1 for linear convolution
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
    
    # Normalization: for uniform heights=1, 
    # integral = n*w = 0.5
    # autoconvolution peak = n*w^2 (for a rectangle of width 0.5, peak of triangle is 0.25)
    # Actually for step function with n intervals of width w, all height 1:
    # The autoconvolution at lag 0 is sum(h_i^2 * w) = n * 1 * w = n*w = 0.5
    # Wait, let me think more carefully.
    # 
    # The step function f(x) = h_i for x in [i*w - 0.25, (i+1)*w - 0.25)
    # Autoconvolution: (f*f)(t) = integral f(x) f(t-x) dx
    # At t=0: sum_i h_i^2 * w = n * 1 * w = 0.5
    # 
    # Normalized max = max(f*f) / (integral(f))^2 * support_length
    # For uniform: 0.5 / (0.5)^2 * 0.5 = 0.5/0.25*0.5 = 1.0
    # 
    # So: score = max_conv / (total * w)^2 * (n * w)
    #       = max_conv * n / (total^2 * w)  ... let me verify
    # integral = total * w
    # score = max_conv / (total*w)^2 * (n*w) = max_conv * n * w / (total^2 * w^2) = max_conv * n / (total^2 * w)
    # For uniform: max_conv = n*w (from discrete sum), total = n
    # score = n*w * n / (n^2 * w) = 1.0 ✓
    
    score = max_conv * n / (total * total * w)
    return score

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
    
    # Gradient of conv w.r.t. f: conv = f * f (linear)
    # d(conv_k)/d(f_i) = f_{k-i} + f_{k-i} ... actually for autoconv:
    # conv_k = sum_i f_i * f_{k-i}
    # d(conv_k)/d(f_i) = f_{k-i} + f_{k-i} = 2*f_{k-i} (when both terms exist)
    # But for the max at peak index p:
    # d(max)/d(f_i) = 2 * f_{p-i} (with appropriate indexing)
    
    peak_idx = np.argmax(conv)
    
    # The gradient of conv[peak_idx] w.r.t. f[i] is 2*f[peak_idx - i] (circular within M)
    # But since we're doing linear conv, we need to be careful.
    # Actually, for linear autoconvolution:
    # conv[t] = sum_i f[i] * f[t-i] where t-i must be in [0, n-1]
    # d(conv[t])/d(f[i]) = f[t-i] if 0 <= t-i < n, else 0 (for the first sum)
    #                     + f[t-i] if 0 <= t-i < n (for the second sum, but it's the same)
    # Wait, conv[t] = sum_i f[i]*f[t-i], so d/d(f[i]) = f[t-i] (from the i term) 
    # but also when j = t-i, d/d(f[i]) picks up f[j] = f[t-i] from the j term
    # So d(conv[t])/d(f[i]) = 2*f[t-i] if 0 <= t-i < n
    
    grad = np.zeros(len(heights), dtype=np.float64)
    for i in range(len(heights)):
        j = peak_idx - i
        if 0 <= j < len(heights):
            grad[i] = 2.0 * heights[j]
    
    return conv, grad, peak_idx

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
    
    # Strategy: Use smooth optimization with log-sum-exp surrogate
    # The objective is max_t (f*f)(t) / (integral(f))^2 * support
    # We minimize this. Use a smooth surrogate: (1/alpha) * log(sum(exp(alpha * conv_t)))
    # to approximate max(conv_t)
    
    # Also, to handle the normalization, we can fix the integral and just minimize max conv
    # Or use a parameterization that keeps things simple.
    
    # Approach 1: Gradient descent on the smooth surrogate
    # Approach 2: Try structured shapes (tapered, plateau, etc.)
    # Approach 3: Combine both
    
    rng = np.random.default_rng(12345)
    
    # First, try some structured shapes quickly
    structured_candidates = []
    
    # Tapered edges (cosine taper)
    for taper_frac in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        h_taper = np.ones(N)
        taper_len = int(taper_frac * N)
        if taper_len > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, taper_len)))
            h_taper[:taper_len] = ramp
            h_taper[-taper_len:] = ramp[::-1]
        structured_candidates.append(h_taper)
    
    # Raised cosine (smooth bump)
    x = np.linspace(0, 1, N)
    for center in [0.5]:
        for width in [0.3, 0.5, 0.7, 0.9, 1.0]:
            h_cos = np.exp(-((x - center) / width) ** 2)
            h_cos = h_cos / np.max(h_cos)  # normalize to max 1
            structured_candidates.append(h_cos)
    
    # Parabolic
    for a in [0.5, 1.0, 2.0, 3.0]:
        h_par = 1.0 - a * (2 * x - 1) ** 2
        h_par = np.maximum(h_par, 0.0)
        if np.max(h_par) > 0:
            structured_candidates.append(h_par)
    
    # Linear taper
    for slope in [0.5, 1.0, 2.0, 3.0, 5.0]:
        h_lin = 1.0 - slope * np.abs(2 * x - 1)
        h_lin = np.maximum(h_lin, 0.0)
        if np.max(h_lin) > 0:
            structured_candidates.append(h_lin)
    
    # Plateau with smooth edges
    for plateau_frac in [0.3, 0.5, 0.7]:
        h_plat = np.ones(N)
        edge_len = int((1 - plateau_frac) * N / 2)
        if edge_len > 1:
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, edge_len)))
            h_plat[:edge_len] = ramp
            h_plat[-edge_len:] = ramp[::-1]
        structured_candidates.append(h_plat)
    
    # Evaluate structured candidates
    for h_cand in structured_candidates:
        if time_left() < 5:
            break
        h_cand = np.maximum(h_cand, 0.0)
        if np.sum(h_cand) <= 0:
            continue
        score = compute_score(h_cand)
        if score < best_score:
            best_score = score
            best_h = h_cand.copy()
    
    # Now do gradient-based optimization from the best candidate
    # Use log-sum-exp smooth surrogate for the max
    
    def smooth_max(conv, alpha):
        """Smooth approximation of max(conv) using log-sum-exp."""
        # Scale to avoid overflow
        m = np.max(conv)
        if m < 0:
            m = 0
        exp_conv = np.exp(alpha * (conv - m))
        return m + np.log(np.sum(exp_conv)) / alpha
    
    def smooth_obj(heights, alpha):
        """Smooth objective: smooth_max(autoconv) / (sum(heights)*w)^2 * n*w"""
        f = np.zeros(M, dtype=np.float64)
        f[:N] = heights
        F = np.fft.rfft(f)
        conv = np.fft.irfft(F * F, n=M)
        
        sm = smooth_max(conv, alpha)
        total = np.sum(heights)
        w = 0.5 / N
        if total <= 1e-10:
            return 1e10
        return sm / (total * w) ** 2 * (N * w)
    
    def smooth_grad(heights, alpha):
        """Gradient of the smooth objective."""
        f = np.zeros(M, dtype=np.float64)
        f[:N] = heights
        F = np.fft.rfft(f)
        conv = np.fft.irfft(F * F, n=M)
        
        # Gradient of smooth_max
        m = np.max(conv)
        if m < 0:
            m = 0
        exp_conv = np.exp(alpha * (conv - m))
        weights = exp_conv / np.sum(exp_conv)  # softmax weights
        
        # d(smooth_max)/d(conv_k) = weights[k]
        # d(conv_k)/d(f_i) = 2*f_{k-i} (for linear autoconv)
        grad_f = np.zeros(M, dtype=np.float64)
        for i in range(N):
            # conv_k = sum_j f_j * f_{k-j}
            # d(conv_k)/d(f_i) = f_{k-i} (from j=i term) + f_{k-i} (from j=k-i term, when i != k-i)
            # Actually: conv_k = sum_j f_j f_{k-j}
            # d/d(f_i) = f_{k-i} (when j=i) + f_{k-i} (when k-j=i, i.e., j=k-i)
            # So d(conv_k)/d(f_i) = f_{k-i} + f_{k-i} = 2*f_{k-i} if 0<=k-i<N
            # But wait, if k-i < 0 or k-i >= N, then f_{k-i} = 0
            # So: d(conv_k)/d(f_i) = 2*f_{k-i} if 0 <= k-i < N, else 0
            
            # Vectorized: for each i, sum over k of weights[k] * 2*f[k-i] (if valid)
            # This is a correlation-like operation
            pass
        
        # More efficient: use FFT
        # grad_f[i] = sum_k weights[k] * d(conv_k)/d(f_i) = sum_k weights[k] * 2*f[k-i] (if 0<=k-i<N)
        # = 2 * (weights * f)[circular correlation]
        # Actually: sum_k weights[k] * f[k-i] = (weights * f) circular correlation at lag i
        # In FFT domain: ifft(fft(weights) * conj(fft(f)))... 
        # Let me just do it directly with FFT
        W = np.fft.rfft(weights[:2*N-1])  # weights is length M, but conv is meaningful in [0, 2N-2]
        # Actually conv has length M, but only [0, 2N-2] is non-zero
        # Let's work with the full M-length arrays
        W_full = np.fft.rfft(weights)
        F_full = np.fft.rfft(f)
        # grad_f[i] = 2 * sum_k weights[k] * f[k-i]
        # = 2 * ifft(W_full * conj(F_full))[i] ... no
        # sum_k weights[k] * f[k-i] = correlation of weights and f at lag i
        # = ifft(fft(weights) * conj(fft(f)))... 
        # Let's use: corr[i] = sum_k weights[k] * f[k-i]
        # In FFT: corr = ifft(fft(weights) * conj(fft(f)))  -- this gives circular correlation
        # But we need linear correlation. Since f is zero-padded to M >= 2N-1, 
        # and weights is also effectively zero outside [0, 2N-2], the circular correlation
        # in M points should equal the linear correlation for i in [0, N-1].
        
        corr = np.fft.irfft(W_full * np.conj(F_full), n=M)
        grad_f = 2.0 * corr[:N]
        
        # Now chain rule: obj = smooth_max / (total*w)^2 * N*w
        # Let S = sum(heights) = total, w = 0.5/N
        # obj = sm / (S*w)^2 * N*w = sm * N / (S^2 * w)
        # d(obj)/d(f_i) = [d(sm)/d(f_i) * N / (S^2 * w)] - [sm * N * 2*S / (S^4 * w) * dS/d(f_i)]
        #               = [grad_f[i] * N / (S^2 * w)] - [sm * 2*N / (S^3 * w) * 1]
        
        S = np.sum(heights)
        w = 0.5 / N
        if S <= 1e-10:
            return np.zeros(N)
        
        sm = smooth_max(conv, alpha)
        obj_grad = grad_f * (N / (S * S * w)) - sm * (2.0 * N / (S * S * S * w))
        
        return obj_grad
    
    # Gradient descent with adaptive alpha
    h_opt = best_h.copy()
    alpha = 50.0  # start with moderate smoothness
    lr = 0.01
    
    for it in range(500):
        if time_left() < 10:
            break
        
        # Compute gradient
        grad = smooth_grad(h_opt, alpha)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        # Update
        h_opt = h_opt - lr * grad
        h_opt = np.maximum(h_opt, 0.0)
        
        # Compute actual score
        score = compute_score(h_opt)
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
        
        # Adapt learning rate and alpha
        if it % 20 == 0:
            # Increase alpha (sharper max approximation)
            alpha = min(alpha * 1.2, 500.0)
            # Decrease learning rate
            lr = max(lr * 0.9, 0.001)
    
    # Try multiple random restarts
    for restart in range(20):
        if time_left() < 15:
            break
        
        # Random smooth initial guess
        freqs = np.arange(N)
        h_init = np.ones(N)
        num_modes = rng.integers(3, 10)
        for _ in range(num_modes):
            k = rng.integers(1, N // 3)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.05, 0.3)
            h_init += amp * np.cos(2 * np.pi * k * freqs / N + phase)
        h_init = np.maximum(h_init, 0.0)
        if np.sum(h_init) <= 0:
            continue
        
        # Optimize from this start
        h_r = h_init.copy()
        alpha_r = 50.0
        lr_r = 0.01
        
        for it in range(200):
            if time_left() < 10:
                break
            
            grad = smooth_grad(h_r, alpha_r)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break
            
            h_r = h_r - lr_r * grad
            h_r = np.maximum(h_r, 0.0)
            
            score = compute_score(h_r)
            if score < best_score:
                best_score = score
                best_h = h_r.copy()
            
            if it % 20 == 0:
                alpha_r = min(alpha_r * 1.15, 500.0)
                lr_r = max(lr_r * 0.92, 0.001)
    
    # Final polish: try fine-tuning the best solution
    if time_left() > 10:
        h_final = best_h.copy()
        alpha_f = 200.0
        lr_f = 0.001
        
        for it in range(100):
            if time_left() < 5:
                break
            
            grad = smooth_grad(h_final, alpha_f)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break
            
            h_final = h_final - lr_f * grad
            h_final = np.maximum(h_final, 0.0)
            
            score = compute_score(h_final)
            if score < best_score:
                best_score = score
                best_h = h_final.copy()
            
            lr_f = max(lr_f * 0.98, 0.0001)
    
    # Ensure non-negativity
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
