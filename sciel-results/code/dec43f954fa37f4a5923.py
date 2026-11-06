import time
import numpy as np

N = 400
_t0 = time.time()
_DEADLINE = 135.0

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _smooth_value_and_grad(h, beta=50.0):
    """
    Compute smooth value and gradient using log-sum-exp approximation of max.
    value = 2N * LSE(R) / S^2
    LSE(R) = max(R) + log(sum(exp(beta*(R - max(R))))) / beta
    """
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf, np.zeros_like(h)
    
    conv = _fft_convolve(h)
    m = np.max(conv)
    # Numerical stability for logsumexp
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    lse = m + np.log(np.sum(np.exp(scaled))) / beta
    
    val = abs(2.0 * N * lse / (s * s))
    
    # Gradient calculation
    # d/dh val = (2N / s^2) * d/dh lse  (ignoring sign for a moment, assuming positive)
    # Actually, val = 2N * lse / s^2.
    # d(val)/dh = (2N * s^2 * d(lse)/dh - 2N * lse * 2s * d(s)/dh) / s^4
    #           = (2N / s^2) * (d(lse)/dh - 2 * lse / s * d(s)/dh)
    
    # d(lse)/dh
    # lse = m + 1/beta * log(sum(exp(beta*(conv-m))))
    # Let p_i = exp(beta*(conv_i - m)) / sum(exp(...))
    # d(lse)/d(conv_i) = p_i
    # d(conv_i)/dh_j = h_{i-j} + h_{i+N-1-j} (for full conv, indices need care)
    
    # Simpler: use numerical gradient for robustness if analytic is too complex to derive quickly.
    # But N=400, numerical gradient is 800 evals. Might be slow.
    # Let's try analytic gradient of LSE.
    
    # p = softmax(beta * (conv - m))
    p = np.exp(scaled) / np.sum(np.exp(scaled))
    
    # d(lse)/dh = sum_k p_k * d(conv_k)/dh
    # conv_k = sum_i h_i h_{i+k}
    # d(conv_k)/dh_j = h_{k-j} + h_{k+N-1-j} ? 
    # Let's check indices.
    # conv[k] = sum_{i} h[i] h[i+k]
    # If we change h[j], terms involving h[j] are:
    # 1. h[j] * h[j+k] (if j+k < N)
    # 2. h[j-k] * h[j] (if j-k >= 0)
    # So d(conv_k)/dh_j = h[j+k] (if valid) + h[j-k] (if valid).
    
    # We need G = sum_k p_k * (h[j+k] + h[j-k])
    # This is a convolution-like operation.
    # G_j = sum_k p_k h_{j+k} + sum_k p_k h_{j-k}
    # Let p_rev = p[::-1].
    # First term is correlation of p and h.
    # Second term is correlation of p_rev and h.
    
    # Using FFT for gradient of LSE
    # grad_lse = ifft(fft(p) * fft(h)) + ifft(fft(p_rev) * fft(h)) ?
    # Note: conv is defined as sum h_i h_{i+k}.
    # In FFT domain, Conv = IFFT(FFT(h) * FFT(h)).
    # d(Conv_k)/dh_j is the kernel of the linear operator.
    
    # Let's use a simpler numerical gradient with a small step, but only if time permits.
    # Given the time limit, I will use a simple coordinate descent or a few steps of gradient descent 
    # with a coarse numerical gradient if analytic is risky.
    
    # Actually, let's just return the value and use a simple random perturbation / hill climbing.
    # This is safer and "new" compared to the complex structured approaches.
    
    return val, None

def construct():
    global _t0
    _t0 = time.time()
    
    # Initialize with ones, then add small noise
    rng = np.random.default_rng(123)
    h = np.ones(N) + 0.01 * rng.normal(0, 1, N)
    
    best_h = h.copy()
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
    
    # Phase 1: Random Restarts and Local Search
    # We will perform random perturbations and keep the best.
    # To be more efficient, we use a "simulated annealing" like schedule.
    
    current_h = best_h.copy()
    current_v = best_v
    
    # Learning rate schedule
    lr = 0.1
    
    for step in range(2000):
        if not deadline_ok():
            break
            
        # Generate perturbation
        # Mix of global and local noise
        noise = rng.normal(0, lr, N)
        trial_h = current_h + noise
        
        trial_v = _value(trial_h)
        
        # Acceptance criterion
        if trial_v < current_v:
            current_h = trial_h
            current_v = trial_v
            update_best(current_h)
            # Slowly decay learning rate
            lr *= 0.999
        else:
            # Occasionally accept worse solutions to escape local minima
            if rng.random() < 0.01:
                current_h = trial_h
                current_v = trial_v
            # Decay learning rate slightly even on rejection
            lr *= 0.9995
            
        # Ensure lr doesn't become too small
        if lr < 1e-6:
            lr = 1e-6

    # Phase 2: Fine-tuning with smaller steps
    lr = 0.01
    for step in range(1000):
        if not deadline_ok():
            break
            
        noise = rng.normal(0, lr, N)
        trial_h = current_h + noise
        trial_v = _value(trial_h)
        
        if trial_v < current_v:
            current_h = trial_h
            current_v = trial_v
            update_best(current_h)
            lr *= 0.99
        else:
            lr *= 0.995
            
        if lr < 1e-8:
            lr = 1e-8

    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    
    return best_h
