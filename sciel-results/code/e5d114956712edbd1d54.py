import time
import numpy as np
from scipy.optimize import minimize

N = 600
DEADLINE = 130.0  # Safety margin before 150s kill

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
    
    # Normalized score: for uniform heights=1, score = 1.0
    score = max_conv * n / (total * total * w)
    return score

def _parametric_heights(params):
    """
    Generate step heights from a parametric representation.
    params = [center, width, taper_exp, amplitude]
    - center: in [0, 1], position of the peak
    - width: in (0, 1), effective width of the bump
    - taper_exp: exponent for the super-ellipse / raised cosine shape
    - amplitude: overall scale
    """
    center, width, taper_exp, amplitude = params
    x = np.linspace(0, 1, N)
    
    # Normalize x relative to center
    t = (x - center) / width
    
    # Clip to avoid extreme values
    t = np.clip(t, -2.0, 2.0)
    
    # Super-ellipse-like shape: (1 - |t|^p)^(1/p) for p > 0
    # For p=1: triangular, p=2: parabolic, p->inf: rectangular
    p = taper_exp
    if p < 0.1:
        p = 0.1
    
    abs_t = np.abs(t)
    mask = abs_t < 1.0
    h = np.zeros(N)
    h[mask] = (1.0 - abs_t[mask] ** p) ** (1.0 / p)
    
    h *= amplitude
    h = np.maximum(h, 0.0)
    
    return h

def _parametric_score(params):
    """Score for the parametric representation."""
    h = _parametric_heights(params)
    if np.sum(h) <= 0:
        return 1e10
    return compute_score(h)

def _parametric_score_and_grad(params):
    """Score and numerical gradient for the parametric representation."""
    h = _parametric_heights(params)
    if np.sum(h) <= 0:
        return 1e10, np.zeros(4)
    
    s = compute_score(h)
    
    # Numerical gradient
    grad = np.zeros(4)
    eps = 1e-6
    for i in range(4):
        params_plus = params.copy()
        params_plus[i] += eps
        s_plus = _parametric_score(params_plus)
        grad[i] = (s_plus - s) / eps
    
    return s, grad

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def time_left():
        return deadline - time.time()
    
    # Start with uniform
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = compute_score(h)
    
    # Phase 1: Grid search over parametric shapes
    if time_left() > 30:
        # Coarse grid
        centers = np.linspace(0.3, 0.7, 5)
        widths = np.linspace(0.2, 0.8, 5)
        tapers = np.array([1.0, 1.5, 2.0, 3.0, 5.0])
        amplitudes = np.array([0.5, 1.0, 1.5, 2.0])
        
        for c in centers:
            if time_left() < 5:
                break
            for wdt in widths:
                if time_left() < 5:
                    break
                for tp in tapers:
                    if time_left() < 5:
                        break
                    for amp in amplitudes:
                        if time_left() < 2:
                            break
                        params = np.array([c, wdt, tp, amp])
                        s = _parametric_score(params)
                        if s < best_score:
                            best_score = s
                            best_h = _parametric_heights(params).copy()
        
        # Finer grid around best
        if time_left() > 10:
            # Find best parametric solution from coarse grid
            best_params = None
            for c in centers:
                for wdt in widths:
                    for tp in tapers:
                        for amp in amplitudes:
                            params = np.array([c, wdt, tp, amp])
                            s = _parametric_score(params)
                            if best_params is None or s < _parametric_score(best_params):
                                best_params = params
            
            if best_params is not None:
                # Local refinement with Nelder-Mead
                for _ in range(3):
                    if time_left() < 5:
                        break
                    res = minimize(_parametric_score, best_params, method='Nelder-Mead',
                                   options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
                    if res.fun < best_score:
                        best_score = res.fun
                        best_h = _parametric_heights(res.x).copy()
                        best_params = res.x
                    else:
                        break
    
    # Phase 2: Gradient-based optimization on full 600-dimensional space
    # Use a smooth max surrogate to make the objective differentiable
    if time_left() > 20:
        M = 1
        while M < 2 * N - 1:
            M *= 2
        
        def smooth_obj_and_grad(heights, alpha=200.0):
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
            
            obj = sm * N / (S * S * w)
            obj_grad = grad_f * (N / (S * S * w)) - sm * (2.0 * N / (S * S * S * w))
            
            return obj, obj_grad
        
        h_opt = best_h.copy()
        momentum = np.zeros(N)
        beta = 0.9
        lr = 0.003
        alpha = 200.0
        
        for it in range(300):
            if time_left() < 5:
                break
            
            obj, grad = smooth_obj_and_grad(h_opt, alpha)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break
            
            momentum = beta * momentum + (1 - beta) * grad
            h_new = h_opt - lr * momentum
            h_new = np.maximum(h_new, 0.0)
            
            score_new = compute_score(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
            
            # Adaptive learning rate
            if score_new >= compute_score(h_opt):
                lr = max(lr * 0.95, 0.0001)
            else:
                lr = min(lr * 1.05, 0.02)
            
            h_opt = h_new.copy()
            
            if it % 30 == 0:
                alpha = min(alpha * 1.2, 1000.0)
    
    # Phase 3: Try a few diverse structured shapes and refine
    if time_left() > 15:
        x = np.linspace(0, 1, N)
        
        candidates = []
        
        # Raised cosine windows with different widths
        for width in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for center in [0.4, 0.5, 0.6]:
                t = (x - center) / width
                mask = np.abs(t) < 1
                h = np.zeros(N)
                h[mask] = 0.5 * (1 + np.cos(np.pi * t[mask]))
                candidates.append(h)
        
        # Super-ellipse shapes
        for p in [1.0, 1.5, 2.0, 3.0]:
            for width in [0.3, 0.5, 0.7]:
                t = (x - 0.5) / width
                mask = np.abs(t) < 1
                h = np.zeros(N)
                h[mask] = (1 - np.abs(t[mask]) ** p) ** (1/p)
                candidates.append(h)
        
        for h_cand in candidates:
            if time_left() < 3:
                break
            h_cand = np.maximum(h_cand, 0.0)
            if np.sum(h_cand) <= 0:
                continue
            score = compute_score(h_cand)
            if score < best_score:
                best_score = score
                best_h = h_cand.copy()
    
    # Final polish
    if time_left() > 10:
        M = 1
        while M < 2 * N - 1:
            M *= 2
        
        def smooth_obj_and_grad(heights, alpha=500.0):
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
            
            W_full = np.fft.rfft(weights)
            F_full = np.fft.rfft(f)
            corr = np.fft.irfft(W_full * np.conj(F_full), n=M)
            grad_f = 2.0 * corr[:N]
            
            S = np.sum(heights)
            w = 0.5 / N
            if S <= 1e-10:
                return 1e10, np.zeros(N)
            
            obj = sm * N / (S * S * w)
            obj_grad = grad_f * (N / (S * S * w)) - sm * (2.0 * N / (S * S * S * w))
            
            return obj, obj_grad
        
        h_final = best_h.copy()
        lr_f = 0.0005
        
        for it in range(100):
            if time_left() < 3:
                break
            
            obj, grad = smooth_obj_and_grad(h_final)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break
            
            h_new = h_final - lr_f * grad
            h_new = np.maximum(h_new, 0.0)
            
            score_new = compute_score(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
                h_final = h_new.copy()
            
            lr_f = max(lr_f * 0.98, 0.00001)
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)

if __name__ == "__main__":
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print("Constructed optimized step function.")
    print(f"Score: {compute_score(np.array(res))}")
