"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 600
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = 140.0  # safety margin under 150s wall-clock
    
    def autoconv(h):
        """Compute linear autoconvolution via FFT."""
        n = len(h)
        n_fft = 2 * n
        H = np.fft.rfft(h, n=n_fft)
        return np.fft.irfft(H * H, n=n_fft)
    
    def score_fn(h):
        """Normalized max of autoconvolution."""
        total = np.sum(h)
        if total < 1e-12:
            return 0.0
        conv = autoconv(h)
        return np.max(conv) / (total * total)
    
    # Strategy: use a symmetric triangular (tent) shape as starting point,
    # then apply coordinate descent + random restarts with different initial shapes.
    
    best_h = None
    best_score = np.inf
    
    def try_optimize(h0, lr0=0.01, max_iter=2000):
        nonlocal best_h, best_score
        h = h0.copy()
        n = len(h)
        total0 = np.sum(h)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            conv = autoconv(h)
            total = np.sum(h)
            if total < 1e-12:
                break
            s = np.max(conv) / (total * total)
            
            if s < best_score:
                best_score = s
                best_h = h.copy()
            
            # Find argmax of conv
            k_max = int(np.argmax(conv))
            
            # Gradient of s w.r.t. h[j]:
            # s = conv[k] / (sum h)^2
            # d conv[k] / d h[j] = h[k-j] + h[j-k] (for linear conv, with bounds)
            # d s / d h[j] = (dconv/dh[j] * total^2 - conv[k] * 2 * total) / total^4
            #             = (dconv/dh[j]) / total^2 - 2*conv[k] / total^3
            
            j = np.arange(n)
            idx1 = k_max - j
            idx2 = j - k_max
            
            dh_dh = np.zeros(n)
            mask1 = (idx1 >= 0) & (idx1 < n)
            mask2 = (idx2 >= 0) & (idx2 < n)
            dh_dh[mask1] += h[idx1[mask1]]
            dh_dh[mask2] += h[idx2[mask2]]
            
            grad = dh_dh / (total * total) - 2.0 * conv[k_max] / (total ** 3)
            
            # Projected gradient step (keep nonnegative)
            h = h - lr0 * grad
            h = np.maximum(h, 0.0)
            
            # Renormalize to keep sum constant (the objective is scale-invariant)
            new_total = np.sum(h)
            if new_total < 1e-12:
                h = h0.copy()
                break
            h = h * (total0 / new_total)
        
        # Final check
        s = score_fn(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Initial guesses: different shapes
    # 1. Uniform
    h_uniform = np.ones(N)
    
    # 2. Triangular (tent) centered
    h_tri = np.abs(np.arange(N) - N // 2)
    h_tri = (N // 2 - h_tri) / (N // 2 + 1)
    h_tri = np.maximum(h_tri, 0)
    
    # 3. Gaussian-like
    x = np.linspace(-1, 1, N)
    h_gauss = np.exp(-4.0 * x * x)
    
    # 4. Cosine bump
    h_cos = np.cos(np.pi * x / 2) ** 2
    h_cos = np.maximum(h_cos, 0)
    
    # 5. Two-peaked
    h_2peak = np.exp(-40*(x - 0.3)**2) + np.exp(-40*(x + 0.3)**2)
    
    candidates = [h_uniform, h_tri, h_gauss, h_cos, h_2peak]
    
    for h0 in candidates:
        if time.time() > deadline:
            break
        try_optimize(h0, lr0=0.01, max_iter=1500)
    
    # Additional round with smaller learning rate for polishing
    if best_h is not None and time.time() < deadline:
        try_optimize(best_h, lr0=0.003, max_iter=1000)
    
    if best_h is None:
        best_h = h_uniform
    
    return list(best_h)
# EVOLVE-BLOCK-END
