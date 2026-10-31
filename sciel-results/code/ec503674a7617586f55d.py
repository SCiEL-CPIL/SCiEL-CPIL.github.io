"""Combined approach: parametric initialization + gradient-based local refinement."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    def compute_metric(h):
        """Compute normalized max of autoconvolution: max(conv) / (sum(h))^2"""
        h = np.asarray(h, dtype=np.float64)
        total = np.sum(h)
        if total < 1e-15:
            return float('inf')
        # Autoconvolution via FFT
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        peak = np.max(acorr)
        return peak / (total * total)
    
    def optimize_from(h_init, max_iters=2000):
        """Optimize using projected gradient descent with adaptive step size."""
        h = h_init.copy()
        best_h = h.copy()
        best_m = compute_metric(h)
        lr = 0.02
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            size = 2 * N
            H = np.fft.rfft(h, n=size)
            acorr = np.fft.irfft(H * np.conj(H), n=size)
            peak_idx = np.argmax(acorr)
            peak = acorr[peak_idx]
            total = np.sum(h)
            
            if peak < 1e-15:
                break
            
            # Gradient of the metric w.r.t. h[j]
            # metric = acorr[peak_idx] / total^2
            # d(acorr[k])/d(h[j]) = h[j-k] + h[j+k] (with bounds) for k != 0
            # d(acorr[0])/d(h[j]) = 2*h[j]
            # d(total^2)/d(h[j]) = 2*total
            
            k = peak_idx
            if k == 0:
                grad_acorr_k = 2.0 * h
            else:
                grad_acorr_k = np.zeros(N)
                # h[j-k] term: valid when j-k in [0, N-1]
                if k < N:
                    grad_acorr_k[k:] += h[:N-k]
                    grad_acorr_k[:N-k] += h[k:]
            
            grad_total_sq = 2.0 * total
            
            # d(metric)/d(h[j]) = (grad_acorr_k[j] * total^2 - acorr[k] * grad_total_sq) / total^4
            #                     = grad_acorr_k[j] / total^2 - acorr[k] * 2 / total^3
            metric_grad = grad_acorr_k / (total * total) - 2.0 * acorr[k] / (total * total * total)
            
            # Gradient step
            h_new = h - lr * metric_grad
            h_new = np.maximum(h_new, 0.0)
            
            m_new = compute_metric(h_new)
            
            if m_new < best_m:
                best_m = m_new
                best_h = h_new.copy()
            
            # Adaptive learning rate
            if m_new < best_m * 0.999:
                lr = min(lr * 1.2, 0.1)
            else:
                lr = max(lr * 0.95, 0.001)
            
            h = h_new
        
        return best_h, best_m
    
    # Generate diverse initializations
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N))
    
    # 2. Cosine window
    initializations.append(np.maximum(np.cos(np.pi * x / 0.25), 0.0))
    
    # 3. Raised cosine (sinc-like)
    initializations.append(np.maximum(0.5 * (1.0 + np.cos(2 * np.pi * x / 0.5)), 0.0))
    
    # 4. Gaussian
    initializations.append(np.exp(-10.0 * x**2))
    
    # 5. Flat top with soft edges
    h_flat = np.ones(N)
    edge_width = 50
    for j in range(edge_width):
        h_flat[j] *= 0.5 * (1.0 - np.cos(np.pi * j / edge_width))
        h_flat[N-1-j] *= 0.5 * (1.0 - np.cos(np.pi * j / edge_width))
    initializations.append(h_flat)
    
    # 6. Two-hump Gaussian
    initializations.append(np.exp(-50.0 * (x - 0.1)**2) + np.exp(-50.0 * (x + 0.1)**2))
    
    # 7. Random smooth
    rng = np.random.RandomState(42)
    smooth = rng.uniform(0.5, 1.5, N)
    from scipy.ndimage import gaussian_filter1d
    smooth = gaussian_filter1d(smooth, sigma=10)
    initializations.append(smooth)
    
    best_h = None
    best_m = float('inf')
    
    for init in initializations:
        if time.time() > deadline:
            break
        
        init = np.maximum(init, 0.0)
        h_opt, m_opt = optimize_from(init, max_iters=1500)
        
        if m_opt < best_m:
            best_m = m_opt
            best_h = h_opt
    
    if best_h is None:
        best_h = np.ones(N)
    
    return list(best_h)

# EVOLVE-BLOCK-END
