import numpy as np
import time

N = 50

def _autoconv_peak(h):
    """Compute the supremum norm of the autoconvolution of h (discrete, circular)."""
    # Autoconvolution: (h * h)(k) = sum_i h[i] * h[(i+k) % N]
    # This is equivalent to the circular convolution of h with h.
    # The peak is the max over all lags k.
    # Using FFT for efficiency:
    H = np.fft.rfft(h, n=2*N)  # linear convolution via zero-padding
    conv = np.fft.irfft(H * H, n=2*N)
    # The circular autoconvolution is related to linear conv.
    # Actually, for circular autoconvolution of length N:
    # C(k) = sum_{i=0}^{N-1} h[i] * h[(i+k) % N]
    # This is the circular convolution of h with h.
    # Let's compute it directly using FFT of length N:
    Hc = np.fft.fft(h, n=N)
    C = np.fft.ifft(Hc * Hc).real
    return np.max(np.abs(C))

def _score(h):
    """Compute the ratio: (sum h_i^2) / (sum h_i * max_autoconv)"""
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    peak = _autoconv_peak(h)
    if peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * peak)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Strategy: Use x-vector parameterization h_i = x_i^2
    # Minimize peak autoconv / (L2^2 / L1) which is equivalent to maximizing score
    # We'll use a stochastic subgradient method with random lag sampling
    
    # Initialize x
    rng = np.random.default_rng(42)
    
    # Try multiple restarts with different initializations
    num_restarts = 20
    steps_per_restart = 2000
    lr = 0.05
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize x: random positive values
        if restart == 0:
            x = np.ones(N)
        elif restart == 1:
            x = np.random.default_rng(restart).random(N) + 0.1
        elif restart == 2:
            # Symmetric pattern
            x = np.sin(np.pi * np.arange(N) / N) + 0.1
        elif restart == 3:
            # Cosine pattern
            x = np.cos(2 * np.pi * np.arange(N) / N) ** 2 + 0.1
        else:
            # Random with varying scales
            scale = 0.5 + rng.random() * 2.0
            x = rng.random(N) * scale + 0.05
        
        x = x.copy()
        current_score = _score(x * x)
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 3.0:
                break
            
            h = x * x
            current_score = _score(h)
            
            if current_score > best_score:
                best_score = current_score
                best_h = h.copy()
            
            # Compute gradient with respect to x
            # score = sum(h^2) / (sum(h) * peak)
            # h_i = x_i^2
            # We want to maximize score, so we do gradient ascent
            
            l1 = np.sum(h)
            l2_sq = np.sum(h * h)
            peak = _autoconv_peak(h)
            
            if l1 < 1e-15 or peak < 1e-15:
                break
            
            # Gradient of score w.r.t. h:
            # d(score)/d(h_i) = [2*h_i * l1 * peak - l2_sq * (d(l1*peak)/d(h_i))] / (l1*peak)^2
            # This is complex. Let's use a simpler approach:
            # Use the fact that we want to minimize peak / (l2_sq / l1)
            # Or equivalently, minimize peak * l1 / l2_sq
            
            # Let's use a direct subgradient approach:
            # Sample a random lag k, compute the contribution to peak
            k = rng.integers(0, N)
            
            # Autoconv at lag k: C(k) = sum_i h[i] * h[(i+k) % N]
            C_k = np.sum(h * np.roll(h, -k))
            
            # Gradient of C_k w.r.t. h_i:
            # dC_k/dh_i = h[(i+k) % N] + h[(i-k) % N] (for i where both terms exist)
            # Actually: C(k) = sum_i h[i] * h[(i+k) % N]
            # dC_k/dh_i = h[(i+k) % N] + sum_j delta(j, i-k) * h[j]... 
            # More carefully: C(k) = sum_i h[i] h[(i+k)%N]
            # dC_k/dh_m = h[(m+k)%N] + h[(m-k)%N] (when m appears in both terms)
            # Wait, let me re-derive:
            # C(k) = sum_i h[i] * h[(i+k) % N]
            # Term involving h[m]: when i=m, we get h[m]*h[(m+k)%N]
            # When (i+k)%N = m, i.e., i = (m-k)%N, we get h[(m-k)%N]*h[m]
            # So dC_k/dh_m = h[(m+k)%N] + h[(m-k)%N]
            
            dC_dh = np.roll(h, -k) + np.roll(h, k)
            
            # Now, score = l2_sq / (l1 * peak)
            # We want to maximize this. Using subgradient:
            # Approximate peak by C_k (stochastic)
            # d(score)/d(h_i) ≈ [2*h_i*l1 - l2_sq * (d(l1)/d(h_i) * C_k + l1 * dC_k/d(h_i))] / (l1*C_k)^2
            # = [2*h_i*l1 - l2_sq*(C_k + l1*dC_k/dh_i)] / (l1*C_k)^2
            
            if C_k < 1e-15:
                continue
            
            grad_h = (2.0 * h * l1 - l2_sq * (C_k + l1 * dC_dh)) / (l1 * C_k) ** 2
            
            # Gradient w.r.t. x: d(h_i)/d(x_i) = 2*x_i
            grad_x = grad_h * 2.0 * x
            
            # Gradient ascent step with adaptive learning rate
            # Use a decay schedule
            lr_step = lr / (1.0 + step * 0.001)
            x = x + lr_step * grad_x
            
            # Ensure x stays positive (project)
            x = np.maximum(x, 0.01)
            
            # Normalize to avoid numerical issues
            x_norm = np.sum(x * x)
            if x_norm > 1e10 or x_norm < 1e-10:
                x = x / np.sqrt(x_norm) * 1.0
        
        # After each restart, check if we improved
        h_final = x * x
        s_final = _score(h_final)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final.copy()
    
    # Final polish: local optimization with smaller steps
    if time.time() < deadline - 10.0:
        x_polish = np.sqrt(best_h)
        for step in range(5000):
            if time.time() > deadline - 5.0:
                break
            
            h = x_polish * x_polish
            l1 = np.sum(h)
            l2_sq = np.sum(h * h)
            peak = _autoconv_peak(h)
            
            if l1 < 1e-15 or peak < 1e-15:
                break
            
            # Use full gradient (average over several lags for stability)
            num_lags = 10
            grad_h = np.zeros(N)
            for _ in range(num_lags):
                k = rng.integers(0, N)
                C_k = np.sum(h * np.roll(h, -k))
                if C_k < 1e-15:
                    continue
                dC_dh = np.roll(h, -k) + np.roll(h, k)
                grad_h += (2.0 * h * l1 - l2_sq * (C_k + l1 * dC_dh)) / (l1 * C_k) ** 2
            grad_h /= num_lags
            
            grad_x = grad_h * 2.0 * x_polish
            
            # Small step
            lr_polish = 0.01 / (1.0 + step * 0.0005)
            x_polish = x_polish + lr_polish * grad_x
            x_polish = np.maximum(x_polish, 0.01)
            
            h_new = x_polish * x_polish
            s_new = _score(h_new)
            if s_new > best_score:
                best_score = s_new
                best_h = h_new.copy()
    
    # Return as list of floats
    return [float(v) for v in best_h]
