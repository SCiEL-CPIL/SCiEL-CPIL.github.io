"""Rewritten program: gradient-based continuous relaxation for signed autocorrelation minimization."""
import numpy as np
import time

N = 400

def _compute_score(x):
    """Compute C3' = max(|autoconv|) / (sum(x))^2 for a signed sequence."""
    if len(x) == 0:
        return np.inf
    s = np.sum(x)
    if s == 0:
        return np.inf
    # Full autoconvolution
    ac = np.convolve(x, x, mode='full')
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)

def _fft_autoconv(x):
    """Compute autoconvolution via FFT for speed."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    ac = np.fft.irfft(X * X, size)[:2 * n - 1]
    return ac

def _soft_max_abs(vals, beta):
    """Soft approximation of max(|vals|) using log-sum-exp."""
    abs_vals = np.abs(vals)
    scaled = beta * abs_vals
    max_val = np.max(scaled)
    if max_val > 500:
        return np.max(abs_vals)
    log_sum = max_val + np.log(np.sum(np.exp(scaled - max_val)))
    return log_sum / beta

def _grad_soft_max_abs(vals, beta):
    """Gradient of soft max(|vals|) w.r.t. vals."""
    abs_vals = np.abs(vals)
    signs = np.sign(vals)
    scaled = beta * abs_vals
    max_val = np.max(scaled)
    if max_val > 500:
        # Hard max case
        max_idx = np.argmax(abs_vals)
        grad = np.zeros_like(vals)
        grad[max_idx] = signs[max_idx]
        return grad
    exp_scaled = np.exp(scaled - max_val)
    weights = exp_scaled / np.sum(exp_scaled)
    grad = beta * weights * signs
    return grad

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill
    
    best_x = np.ones(N)
    best_score = _compute_score(best_x)
    
    # Try a few random initializations to see if we can beat the trivial solution
    rng = np.random.RandomState(42)
    
    # First, let's try the M-sequence (m-sequences / maximal length sequences)
    # which have good autocorrelation properties
    # For N=400, we can use a linear congruential generator based approach
    
    # Actually, let's use a different approach: 
    # The key insight is that for the all-ones sequence:
    # sum = N, max|ac| = N, so C3' = N/N^2 = 1/N = 0.0025
    # 
    # Can we do better? We need max|ac| / (sum)^2 < 1/N
    # i.e., max|ac| < sum^2 / N
    #
    # If we use a sequence with sum = 0 (balanced), then C3' is undefined/infinite
    # So we need |sum| to be large relative to max|ac|.
    #
    # For a ±1 sequence, sum is at most N (all ones) or at least -N.
    # The all-ones sequence achieves sum=N and max|ac|=N.
    #
    # What if we use a sequence with sum close to N but with lower peak autocorrelation?
    # For example, if we flip a few -1's, sum decreases slightly but the peak 
    # autocorrelation might decrease more.
    #
    # Let's try: start with all ones, then use gradient descent on the continuous relaxation.
    
    # Continuous relaxation: let x_i be real in [-1, 1], minimize soft_max|ac(x)| / (sum(x))^2
    # Then round to ±1.
    
    x = np.ones(N, dtype=np.float64)
    lr = 0.05
    beta = 50.0  # Temperature for soft max
    
    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
    
    for step in range(2000):
        if time.time() - start_time > deadline - 5:
            break
        
        # Compute autoconvolution via FFT
        X = np.fft.rfft(x, fft_size)
        ac = np.fft.irfft(X * X, fft_size)[:2 * N - 1]
        
        s = np.sum(x)
        if abs(s) < 1e-6:
            # Bad state, reset
            x = np.ones(N, dtype=np.float64)
            continue
        
        # Objective: soft_max|ac| / s^2
        # Gradient w.r.t. x:
        # d/dx [soft_max|ac| / s^2] = (grad_soft_max * s^2 - soft_max * 2*s * 1) / s^4
        #                            = grad_soft_max / s^2 - 2 * soft_max / s^3
        #                            = (1/s^2) * (grad_soft_max - 2*soft_max/s)
        
        soft_max_val = _soft_max_abs(ac, beta)
        grad_ac = _grad_soft_max_abs(ac, beta)
        
        # Gradient of autoconvolution w.r.t. x
        # ac_k = sum_i x_i x_{i+k} (for k>=0)
        # d ac_k / d x_j = x_{j-k} + x_{j+k} (with appropriate bounds)
        # This is equivalent to: grad_x = 2 * (x * ac) in the convolution sense
        # More precisely, if ac = convolve(x, x), then d ac / d x is a linear operator
        # that can be computed via FFT:
        # ac = ifft(fft(x) * fft(x))
        # d ac / d x_j = 2 * ifft(fft(x) * e_j) where e_j is the j-th basis vector
        # In matrix form, this is 2 * circulant matrix formed from x
        # Using FFT: grad_x = 2 * ifft(fft(x) * fft(grad_ac_reversed))
        # Actually, the gradient of sum_k ac_k * g_k w.r.t. x is:
        # 2 * ifft(fft(x) * fft(g_reversed))
        
        # The autoconvolution ac_k = sum_i x_i x_{i+k}
        # Let's define g_k = grad_ac_k (the soft max gradient)
        # Then d/dx_j [sum_k g_k ac_k] = sum_k g_k * (x_{j-k} + x_{j+k})
        # This is 2 * (g * x) where * denotes the appropriate cross-correlation
        
        # Using FFT: 
        # ac = ifft(|fft(x)|^2)
        # d ac_k / d x_j: the Jacobian is a Toeplitz/Hankel structure
        # Easier: use the fact that if we define f(x) = sum_k g_k * ac_k
        # then grad_x f = 2 * ifft(fft(x) * fft(g_flip))
        # where g_flip[k] = g_{2N-2-k} (reversal)
        
        g_flip = grad_ac[::-1].copy()
        G = np.fft.rfft(g_flip, fft_size)
        grad_x = 2.0 * np.fft.irfft(X * G, fft_size)[:N]
        
        # Now combine with the quotient rule
        obj_grad = grad_x / (s * s) - 2.0 * soft_max_val * np.ones(N) / (s * s * s)
        
        # Gradient descent
        x -= lr * obj_grad
        
        # Project to [-1, 1]
        x = np.clip(x, -1.0, 1.0)
        
        # Periodically round and check
        if step % 50 == 0:
            x_rounded = np.where(x >= 0, 1.0, -1.0)
            score = _compute_score(x_rounded)
            if score < best_score:
                best_score = score
                best_x = x_rounded.copy()
        
        # Adaptive learning rate
        if step > 500:
            lr = 0.01
        elif step > 1500:
            lr = 0.003
    
    # Final check with rounded version
    x_rounded = np.where(x >= 0, 1.0, -1.0)
    final_score = _compute_score(x_rounded)
    if final_score < best_score:
        best_score = final_score
        best_x = x_rounded.copy()
    
    # Also try a few random ±1 sequences for comparison
    for trial in range(10):
        if time.time() - start_time > deadline - 2:
            break
        x_rand = rng.choice([-1.0, 1.0], size=N)
        s = np.sum(x_rand)
        if abs(s) < N * 0.1:  # Skip if too unbalanced
            continue
        score = _compute_score(x_rand)
        if score < best_score:
            best_score = score
            best_x = x_rand.copy()
    
    # Return as integer-like sequence
    return np.where(best_x >= 0, 1, -1)

# EVOLVE-BLOCK-END
