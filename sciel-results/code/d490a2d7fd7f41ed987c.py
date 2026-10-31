"""Optimized program: minimize max|autoconv| / (sum(x))^2 via gradient-based real-valued search."""
import numpy as np
import time

N = 400

def _autoconv_full(x):
    """Compute full autoconvolution of x using FFT."""
    n = len(x)
    size = 2 * n
    fft_x = np.fft.fft(x, size)
    acov = np.fft.ifft(fft_x * np.conj(fft_x)).real
    return acov

def _autoconv_max_abs(x):
    """Compute max absolute value of full autoconvolution of x."""
    acov = _autoconv_full(x)
    return np.max(np.abs(acov))

def _score(x):
    """Compute the C3' ratio: max|autoconv| / (sum(x))^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _smooth_objective(x, lam=0.1):
    """Smooth approximation of max|autoconv| using log-sum-exp."""
    acov = _autoconv_full(x)
    abs_acov = np.abs(acov)
    # Use a smooth max approximation
    scaled = abs_acov / lam
    max_scaled = np.max(scaled)
    if max_scaled > 100:
        # Avoid overflow
        return lam * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    return lam * np.log(np.sum(np.exp(scaled)))

def _gradient_smooth(x, lam=0.1):
    """Gradient of the smooth objective with respect to x."""
    acov = _autoconv_full(x)
    abs_acov = np.abs(acov)
    
    # Compute weights for the log-sum-exp gradient
    scaled = abs_acov / lam
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)
    
    # d/dx of |acov[k]|: need to be careful with the sign
    # acov[k] = sum_i x[i] * x[i+k] for k >= 0
    # acov[k] = sum_i x[i] * x[i-k] for k < 0 (stored at index k+n)
    
    n = len(x)
    grad = np.zeros(n)
    
    # For each lag k, acov[k] contributes to grad
    # d(acov[k])/dx[j] = x[j+k] + x[j-k] (with appropriate bounds)
    
    for k in range(n):
        # Positive lag k: acov[k] = sum_i x[i]*x[i+k]
        if abs_acov[k] > 0:
            sign_k = np.sign(acov[k])
        else:
            sign_k = 1.0
        
        w = weights[k] * sign_k
        # d(acov[k])/dx[j] for j where both x[j] and x[j+k] exist
        # x[j] appears in acov[k] when paired with x[j+k] (if j+k < n)
        # and when paired with x[j-k] (if j-k >= 0)
        
        # Contribution from x[j] * x[j+k]: j ranges 0 to n-1-k
        # Contribution from x[j] * x[j-k]: j ranges k to n-1
        
        if k == 0:
            # acov[0] = sum_i x[i]^2, d/dx[j] = 2*x[j]
            grad += w * 2 * x
        else:
            # x[j] paired with x[j+k] for j in [0, n-1-k]
            if n - 1 - k >= 0:
                grad[:n-k] += w * x[k:]
            # x[j] paired with x[j-k] for j in [k, n-1]
            grad[k:] += w * x[:n-k]
    
    # Also handle negative lags stored at indices n to 2n-1
    # acov[n+k] = acov[k] for the negative lag -k (by symmetry of autoconvolution)
    # Actually, for real sequences, acov is symmetric: acov[k] = acov[-k]
    # So the full autoconvolution from FFT has acov[0:n] and acov[n:2n-1]
    # acov[n+k] = sum_i x[i]*x[i-k] = acov[k] (same as positive lag)
    
    # Wait, let me reconsider. The FFT autoconvolution gives:
    # acov[k] = sum_i x[i] * x[i+k] for k = 0, 1, ..., n-1
    # acov[n-k] = sum_i x[i] * x[i-k] for k = 1, 2, ..., n-1
    # These are equal by symmetry for real x.
    
    # So we only need to process k = 0 to n-1, and the negative lags are redundant.
    
    # But we included them in the log-sum-exp. Let me fix this.
    # Actually, the full autoconvolution from FFT of size 2n gives 2n values.
    # For a real sequence of length n, the circular autoconvolution would be different.
    # Let me just use the first n values (positive lags) for the objective.
    
    # Recompute using only positive lags
    abs_acov_pos = np.abs(acov[:n])
    scaled_pos = abs_acov_pos / lam
    max_scaled_pos = np.max(scaled_pos)
    exp_scaled_pos = np.exp(scaled_pos - max_scaled_pos)
    weights_pos = exp_scaled_pos / np.sum(exp_scaled_pos)
    
    grad = np.zeros(n)
    for k in range(n):
        if abs_acov_pos[k] > 0:
            sign_k = np.sign(acov[k])
        else:
            sign_k = 1.0
        
        w = weights_pos[k] * sign_k
        
        if k == 0:
            grad += w * 2 * x
        else:
            if n - 1 - k >= 0:
                grad[:n-k] += w * x[k:]
            grad[k:] += w * x[:n-k]
    
    return grad

def _project_to_bounds(x, lo=-1.0, hi=1.0):
    """Project x to the box [lo, hi]."""
    return np.clip(x, lo, hi)

def _gradient_descent(x_init, lr=0.01, max_iters=5000, lam=0.1, deadline=None):
    """Simple gradient descent on the smooth objective."""
    x = x_init.copy()
    best_x = x.copy()
    best_val = _score(x)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient
        g = _gradient_smooth(x, lam)
        
        # Take a step
        x_new = x - lr * g
        x_new = _project_to_bounds(x_new)
        
        # Evaluate
        val_new = _score(x_new)
        
        if val_new < best_val:
            best_val = val_new
            best_x = x_new.copy()
        
        # Adaptive learning rate
        if np.isfinite(val_new):
            # Simple backtracking if we get stuck
            if it % 100 == 0 and it > 0:
                # Try reducing learning rate if no progress
                pass
        
        x = x_new
    
    return best_x, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    rng = np.random.RandomState(42)
    
    # Initialize with sinusoidal profile
    t = np.linspace(0, 1, N)
    candidates = []
    
    # All ones (baseline)
    candidates.append(np.ones(N))
    
    # Sinusoidal profiles with different frequencies and phases
    for freq in [1, 2, 3, 4, 5, 8, 10]:
        for phase in [0, np.pi/2, np.pi]:
            x_sin = np.sin(2 * np.pi * freq * t + phase)
            # Normalize to [-1, 1]
            x_sin = x_sin / np.max(np.abs(x_sin))
            candidates.append(x_sin.copy())
    
    # Random smooth profiles
    for trial in range(5):
        # Generate a smooth random profile using low-frequency Fourier components
        coeffs = rng.randn(10) * np.exp(-np.arange(10) * 0.3)
        x_rand = np.zeros(N)
        for k, c in enumerate(coeffs):
            x_rand += c * np.cos(2 * np.pi * (k+1) * t)
        x_rand = x_rand / np.max(np.abs(x_rand))
        candidates.append(x_rand.copy())
    
    # Run gradient descent on each candidate
    best_x = None
    best_val = float('inf')
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 5:
            break
        
        # Allocate remaining time proportionally
        remaining = deadline - time.time()
        n_remaining = len(candidates) - i
        time_per_cand = remaining / n_remaining
        
        max_iters = int(time_per_cand * 100)  # Rough estimate
        max_iters = max(100, min(max_iters, 5000))
        
        x_opt, val_opt = _gradient_descent(
            cand, lr=0.05, max_iters=max_iters, lam=0.05, 
            deadline=time.time() + time_per_cand
        )
        
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
    
    # If we have time, do more iterations on the best
    if best_x is not None and time.time() < deadline - 5:
        remaining = deadline - time.time()
        max_iters = int(remaining * 50)
        max_iters = max(500, min(max_iters, 20000))
        
        x_opt, val_opt = _gradient_descent(
            best_x, lr=0.01, max_iters=max_iters, lam=0.05,
            deadline=deadline
        )
        
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
    
    return best_x

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
