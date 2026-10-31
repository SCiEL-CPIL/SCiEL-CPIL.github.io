import numpy as np
import time

N = 400

def _autoconv(x):
    """Compute full autoconvolution of x via FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    ac = np.fft.irfft(X * np.conj(X), size)[:2 * n - 1]
    return ac

def _score(x):
    """Compute max|autoconv| / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv(x)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Baseline: All ones
    x_best = np.ones(N, dtype=np.float64)
    best_score = _score(x_best)
    
    # Try to improve using gradient descent on continuous values
    # We fix sum(x) = 1 to remove scale invariance
    # Minimize max(|ac|)
    
    x = np.ones(N, dtype=np.float64)
    x /= np.sum(x)  # sum is 1
    
    rng = np.random.default_rng(42)
    
    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
        
    # Use smooth max surrogate: softmax of |ac| with temperature
    # f(x) = (1/T) * log(sum(exp(T * |ac_k|)))
    # This gives a smooth approximation to max(|ac|)
    
    T = 50.0  # Temperature for soft-max (higher = sharper)
    
    def _smooth_max_abs_ac(ac):
        """Compute smooth max of |ac| and its gradient w.r.t. ac."""
        abs_ac = np.abs(ac)
        # Shift for numerical stability
        m = np.max(abs_ac)
        shifted = T * (abs_ac - m)
        # Clip to avoid overflow
        shifted = np.clip(shifted, -100, 100)
        exp_shifted = np.exp(shifted)
        log_sum = np.log(np.sum(exp_shifted))
        value = m + log_sum / T
        # Gradient w.r.t. ac: softmax weights * sign(ac)
        weights = exp_shifted / np.sum(exp_shifted)
        grad_ac = weights * np.sign(ac)
        return value, grad_ac
    
    def _grad_smooth_max(x):
        """Compute gradient of smooth max |ac| w.r.t. x using FFT."""
        ac = _autoconv(x)
        value, grad_ac = _smooth_max_abs_ac(ac)
        
        # Gradient of ac[k] w.r.t. x[j]:
        # ac[k] = sum_i x[i]*x[k-i]
        # d(ac[k])/dx[j] = x[k-j] + x[j-k] (when indices valid)
        # 
        # In frequency domain: 
        # ac = ifft(fft(x)^2)
        # d(ac)/dx = ifft(2*fft(x)*fft(dx))
        # 
        # So grad_x[j] = sum_k grad_ac[k] * d(ac[k])/dx[j]
        #              = sum_k grad_ac[k] * (x[k-j] + x[j-k])
        #
        # This is a convolution: grad_x = convolve(grad_ac, x) + convolve(grad_ac, x_reversed)
        # But we can compute it via FFT:
        # 
        # Let's compute directly using the FFT relation:
        # ac = ifft(fft(x) * fft(x))
        # d(ac)/dx = ifft(2 * fft(x) * fft(dx))
        # 
        # So if we want grad_x = d(smooth_max)/dx, and we have grad_ac = d(smooth_max)/dac,
        # then: grad_x = ifft(2 * fft(x) * fft(grad_ac_padded))
        # where grad_ac_padded is grad_ac zero-padded to fft_size
        
        # Pad grad_ac to fft_size
        grad_ac_padded = np.zeros(fft_size, dtype=np.float64)
        grad_ac_padded[:len(grad_ac)] = grad_ac
        
        # Compute gradient via FFT
        X = np.fft.rfft(x, fft_size)
        G = np.fft.rfft(grad_ac_padded, fft_size)
        grad_x = np.fft.irfft(2 * X * G, fft_size)[:N]
        
        return value, grad_x
    
    step_size = 0.005
    no_improve = 0
    max_no_improve = 100
    
    for iter in range(8000):
        if time.time() > deadline - 10:
            break
            
        value, grad_x = _grad_smooth_max(x)
        
        # Current actual score
        s = np.sum(x)
        ac = _autoconv(x)
        max_ac = np.max(np.abs(ac))
        cur_score = max_ac / (s * s)
        
        if cur_score < best_score:
            best_score = cur_score
            x_best = x.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
        
        # Normalize gradient to have unit norm for stable updates
        grad_norm = np.linalg.norm(grad_x)
        if grad_norm < 1e-15:
            break
        grad_normed = grad_x / grad_norm
        
        # Update x in negative gradient direction
        x_new = x - step_size * grad_normed
        
        # Enforce sum(x_new) = 1
        x_new = x_new - np.mean(x_new) + 1.0 / N
        
        x = x_new
        
        # Adaptive step size: decrease if oscillating
        if no_improve > 30:
            step_size *= 0.8
    
    # Final check
    final_score = _score(x_best)
    if final_score < best_score:
        best_score = final_score

    return x_best
