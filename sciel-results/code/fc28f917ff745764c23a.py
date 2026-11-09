"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _autoconv_max_abs(x):
    """Compute the maximum absolute value of the linear autoconvolution of x."""
    # Use FFT for efficient convolution
    n = len(x)
    fsize = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, fsize)
    conv = np.fft.irfft(X * X, fsize)[:2 * n - 1]
    return np.max(np.abs(conv))

def _score(x):
    """Compute the C3' score: max_abs_autoconv / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if s == 0:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _random_signed_step(n):
    """Generate a random signed step function of length n with values in {-1, 0, 1}."""
    # Random blocks of constant sign
    x = np.zeros(n, dtype=np.float64)
    pos = 0
    while pos < n:
        # Random block length
        blen = np.random.randint(1, max(2, n // 50))
        blen = min(blen, n - pos)
        # Random sign: -1, 0, or 1
        val = np.random.choice([-1.0, 0.0, 1.0], p=[0.4, 0.2, 0.4])
        x[pos:pos + blen] = val
        pos += blen
    return x

def _sign_flip_improve(x, max_iters=50000):
    """Local search: flip signs of individual elements to improve score."""
    best_score = _score(x)
    x = x.copy()
    n = len(x)
    
    # Precompute initial autoconv
    fsize = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, fsize)
    conv = np.fft.irfft(X * X, fsize)[:2 * n - 1]
    s = np.sum(x)
    
    for _ in range(max_iters):
        improved = False
        # Try flipping a random position
        for trial in range(20):
            i = np.random.randint(n)
            if x[i] == 0:
                continue
            # Flip sign
            old_val = x[i]
            x[i] = -old_val
            new_s = s - 2 * old_val
            if new_s == 0:
                x[i] = old_val
                continue
            
            # Recompute autoconv efficiently (full recompute for simplicity)
            X_new = np.fft.rfft(x, fsize)
            conv_new = np.fft.irfft(X_new * X_new, fsize)[:2 * n - 1]
            new_score = np.max(np.abs(conv_new)) / (new_s * new_s)
            
            if new_score < best_score:
                best_score = new_score
                s = new_s
                improved = True
                break
            else:
                x[i] = old_val
        
        if not improved:
            break
    
    return x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_x = None
    best_score = float('inf')
    
    # Phase 1: Generate diverse initial candidates
    n_restarts = 0
    max_restarts = 50
    
    while n_restarts < max_restarts:
        if time.time() > deadline - 30.0:
            break
        
        # Generate random signed step function
        x = _random_signed_step(N)
        
        # Ensure non-zero sum
        if np.sum(x) == 0:
            # Add a small bias
            x[0] = 1.0 if x[0] != 1.0 else -1.0
        
        score = _score(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
        
        n_restarts += 1
    
    # Phase 2: Local sign-flip optimization on best candidate
    if best_x is not None and time.time() < deadline - 10.0:
        best_x, best_score = _sign_flip_improve(best_x, max_iters=30000)
    
    # Phase 3: Try a few more restarts with refinement if time permits
    while time.time() < deadline - 5.0:
        x = _random_signed_step(N)
        if np.sum(x) == 0:
            x[0] = 1.0 if x[0] != 1.0 else -1.0
        
        x, score = _sign_flip_improve(x, max_iters=5000)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
    
    # Return the best signed step function found
    if best_x is None:
        # Fallback: return ones
        return np.ones(N)
    
    # Round to clean signed values
    return np.sign(best_x)

# EVOLVE-BLOCK-END
