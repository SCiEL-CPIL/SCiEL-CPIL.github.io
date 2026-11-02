import time
import numpy as np
from scipy.optimize import minimize, linprog

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
    
    peak_idx = np.argmax(conv)
    
    # Efficient gradient using FFT-based correlation
    # grad[i] = 2 * f[peak_idx - i] if 0 <= peak_idx - i < N, else 0
    # This is a flipped and shifted version of f
    n = len(heights)
    grad = np.zeros(n, dtype=np.float64)
    
    # For each i, j = peak_idx - i must be in [0, n-1]
    # So i must be in [peak_idx - n + 1, peak_idx] intersect [0, n-1]
    i_min = max(0, peak_idx - n + 1)
    i_max = min(n - 1, peak_idx)
    if i_min <= i_max:
        indices = np.arange(i_min, i_max + 1)
        j_indices = peak_idx - indices
        grad[indices] = 2.0 * heights[j_indices]
    
    return conv, grad, peak_idx

def _parametric_tapered_heights(params):
    """Generate step heights from parametric top-hat with tapered edges.
    
    params: [center, half_width, taper_power, taper_depth]
    - center: center position in [0, 1] (normalized)
    - half_width: half-width in [0, 0.5] (normalized)
    - taper_power: exponent for super-ellipse taper (1=triangle, 2=parabola, inf=rectangle)
    - taper_depth: fraction of width that is tapered [0, 1]
    """
    center, half_width, taper_power, taper_depth = params
    
    # Clamp parameters to valid ranges
    center = np.clip(center, 0.0, 1.0)
    half_width = np.clip(half_width, 0.05, 0.5)
    taper_power = np.clip(taper_power, 1.0, 50.0)
    taper_depth = np.clip(taper_depth, 0.0, 1.0)
    
    x = np.linspace(0, 1, N)
    
    # Normalized distance from center, scaled by half_width
    dist = np.abs(x - center) / half_width
    
    # Tapered top-hat: 1 in center, tapers to 0 at edges
    # For |dist| < 1-taper_depth: height = 1
    # For 1-taper_depth <= |dist| <= 1: height = (1 - (|dist|-(1-taper_depth))/taper_depth)^taper_power
    # For |dist| > 1: height = 0
    
    h = np.zeros(N)
    inner_mask = dist < (1 - taper_depth)
    h[inner_mask] = 1.0
    
    outer_mask = (dist >= (1 - taper_depth)) & (dist <= 1.0)
    if np.any(outer_mask):
        t = (dist[outer_mask] - (1 - taper_depth)) / taper_depth
        t = np.clip(t, 0, 1)
        h[outer_mask] = (1 - t) ** taper_power
    
    # Ensure non-negative
    h = np.maximum(h, 0.0)
    
    return h

def _parametric_optimize(start_time, deadline):
    """Optimize parametric top-hat with tapered edges using Nelder-Mead."""
    def time_left():
        return deadline - time.time()
    
    best_score = np.inf
    best_h = None
    
    # Initial parameter guesses
    initial_params = [
        [0.5, 0.4, 2.0, 0.3],   # Centered, medium width, parabolic taper
        [0.5, 0.35, 1.5, 0.4],  # Narrower, steeper taper
        [0.5, 0.45, 3.0, 0.2],  # Wider, gentler taper
        [0.5, 0.3, 2.0, 0.5],   # Narrow, deep taper
        [0.5, 0.4, 1.0, 0.5],   # Triangular taper
        [0.5, 0.45, 2.0, 0.1],  # Nearly rectangular with slight taper
    ]
    
    for init in initial_params:
        if time_left() < 15:
            break
        
        def objective(params):
            h = _parametric_tapered_heights(params)
            if np.sum(h) < 1e-10:
                return 1e10
            return compute_score(h)
        
        try:
            result = minimize(
                objective,
                init,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_h = _parametric_tapered_heights(result.x)
        except Exception:
            pass
    
    # Also try a grid search for global coverage
    if time_left() > 20:
        for center in [0.5]:  # Symmetric only
            for half_width in np.linspace(0.25, 0.5, 10):
                for taper_power in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
                    for taper_depth in np.linspace(0.05, 0.6, 10):
                        if time_left() < 5:
                            return best_score, best_h
                        
                        h = _parametric_tapered_heights([center, half_width, taper_power, taper_depth])
                        if np.sum(h) < 1e-10:
                            continue
                        
                        s = compute_score(h)
                        if s < best_score:
                            best_score = s
                            best_h = h.copy()
    
    return best_score, best_h

def _mirror_descent_optimize(heights, start_time, deadline, M):
    """Mirror Descent (Multiplicative Weights) optimization for nonnegative heights."""
    def time_left():
        return deadline - time.time()
    
    w = 0.5 / N
    h = heights.copy()
    best_score = compute_score(h)
    best_h = h.copy()
    
    # Multiplicative weights / Mirror Descent
    # Update: h_new[i] = h[i] * exp(-eta * grad[i])
    # Then project to nonnegative and normalize
    
    eta = 0.05  # Learning rate
    for it in range(500):
        if time_left() < 10:
            break
        
        f = np.zeros(M, dtype=np.float64)
        f[:N] = h
        F = np.fft.rfft(f)
        conv = np.fft.irfft(F * F, n=M)
        
        # Find peak and compute gradient
        peak_idx = np.argmax(conv)
        grad = np.zeros(N, dtype=np.float64)
        
        i_min = max(0, peak_idx - N + 1)
        i_max = min(N - 1, peak_idx)
        if i_min <= i_max:
            indices = np.arange(i_min, i_max + 1)
            j_indices = peak_idx - indices
            grad[indices] = 2.0 * h[j_indices]
        
        S = np.sum(h)
        if S <= 1e-10:
            break
        
        # Normalize gradient for stable updates
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        grad = grad / grad_norm
        
        # Multiplicative update
        h = h * np.exp(-eta * grad)
        h = np.maximum(h, 0.0)
        
        # Normalize to maintain reasonable scale
        if np.sum(h) > 0:
            h = h / np.sum(h) * N
        
        score = compute_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
        
        # Decay learning rate
        eta = max(eta * 0.995, 0.001)
    
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
    
    # Phase 1: Parametric tapered top-hat optimization
    if time_left() > 30:
        s1, h1 = _parametric_optimize(start_time, deadline)
        if h1 is not None and s1 < best_score:
            best_score = s1
            best_h = h1.copy()
    
    # Phase 2: Structured shapes
    if time_left() > 15:
        x = np.linspace(0, 1, N)
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
        
        # Gaussian bump
        for width in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            h_gauss = np.exp(-((x - 0.5) / width) ** 2)
            structured_candidates.append(h_gauss)
        
        # Parabolic
        for a in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h_par = 1.0 - a * (2 * x - 1) ** 2
            h_par = np.maximum(h_par, 0.0)
            if np.max(h_par) > 0:
                structured_candidates.append(h_par)
        
        # Linear taper
        for slope in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h_lin = 1.0 - slope * np.abs(2 * x - 1)
            h_lin = np.maximum(h_lin, 0.0)
            if np.max(h_lin) > 0:
                structured_candidates.append(h_lin)
        
        # Plateau with smooth edges
        for plateau_frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
            h_plat = np.ones(N)
            edge_len = int((1 - plateau_frac) * N / 2)
            if edge_len > 1:
                ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, edge_len)))
                h_plat[:edge_len] = ramp
                h_plat[-edge_len:] = ramp[::-1]
            structured_candidates.append(h_plat)
        
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
    
    # Phase 3: Mirror Descent optimization
    if time_left() > 25:
        s3, h3 = _mirror_descent_optimize(best_h, start_time, deadline, M)
        if h3 is not None and s3 < best_score:
            best_score = s3
            best_h = h3.copy()
    
    # Phase 4: Gradient-based optimization with smooth surrogate (refined)
    if time_left() > 20:
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
            
            sm = m + np.log(np.sum(exp_conv)) / alpha
            
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
    
    # Final polish: high-precision gradient refinement
    if time_left() > 10:
        h_final = best_h.copy()
        alpha_f = 200.0
        lr_f = 0.001
        
        for it in range(100):
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
