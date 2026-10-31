"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _local_sign_flip(h, max_passes=5, deadline=None):
    """Local optimization: try flipping each element's sign, keep if improves."""
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        if not improved:
            break
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Initialize with a raised cosine (Hanning) window - structurally smoother
    # than rectangular, reducing boundary autoconvolution peaks
    x = np.arange(N)
    h_init = 0.5 * (1 + np.cos(2 * np.pi * x / (N - 1)))
    # Scale to have reasonable magnitude
    h_init = h_init * 2.0  # max value 2, min value 0, but we want signed
    
    # Actually, let's think about this differently. The task says "signed step heights"
    # so values can be any real numbers, not just ±1. Let's use a smooth window
    # centered around zero with positive sum.
    
    # Use a raised cosine window that's positive, then optimize
    h = 0.5 * (1 + np.cos(2 * np.pi * x / N))
    # This has sum ~ N/2, and smooth boundaries
    
    best_h = h.copy()
    best_val = _score(best_h)
    
    # Also try ones(N) as baseline
    h_ones = np.ones(N)
    val_ones = _score(h_ones)
    if val_ones < best_val:
        best_val = val_ones
        best_h = h_ones.copy()
    
    # Local optimization from the tapered window
    h_opt, val_opt = _local_sign_flip(h, max_passes=3, deadline=deadline)
    if val_opt < best_val:
        best_val = val_opt
        best_h = h_opt
    
    # Try multiple random initializations with local optimization
    np.random.seed(42)
    n_restarts = 50
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        if r % 5 == 0:
            # Random ±1
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 5 == 1:
            # Mostly +1 with some random -1
            h = np.ones(N)
            n_neg = np.random.randint(N//4, N//2)
            neg_idx = np.random.choice(N, n_neg, replace=False)
            h[neg_idx] = -1.0
        elif r % 5 == 2:
            # Chirp-like pattern
            xx = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * xx * (xx+1) / (2*N)))
            h[h == 0] = 1.0
        elif r % 5 == 3:
            # Smooth raised cosine with random phase
            phase = np.random.uniform(0, 2*np.pi)
            h = 0.5 * (1 + np.cos(2 * np.pi * x / N + phase))
        else:
            # Block pattern
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(1, N//10 + 1)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos+block_len] = sign
                pos += block_len
        
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            h = np.sign(h + 0.5)
        
        h_opt, val_opt = _local_sign_flip(h, max_passes=3, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
    
    return best_h

# EVOLVE-BLOCK-END
