import numpy as np
import time
import scipy.signal
import scipy.fft

N = 400

# EVOLVE-BLOCK-START

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
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

def _local_sign_flip(h, max_passes=20, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves.
    Uses FFT for simplicity and correctness, optimized for N=400.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    # Precompute FFT of h
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    max_abs_ac = np.max(np.abs(ac))
    s = np.sum(h)
    if abs(s) < 1e-6:
        return h, 1e10
    best_val = max_abs_ac / (s * s)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            # Try flipping sign at position i
            v = h[i]
            delta = -2.0 * v  # If v=1, delta=-2 (new -1). If v=-1, delta=2 (new 1).
            
            # Update h
            h[i] = h[i] + delta
            
            # Recompute FFT and autoconv
            # Optimization: Only recompute if necessary? No, FFT is fast enough.
            H_new = np.fft.rfft(h, fsize)
            ac_new = np.fft.irfft(H_new * H_new, fsize)[:size]
            max_abs_new = np.max(np.abs(ac_new))
            s_new = s + delta
            
            if abs(s_new) < 1e-6:
                h[i] = v  # revert
                continue
                
            val_new = max_abs_new / (s_new * s_new)
            
            if val_new < best_val - 1e-12:
                best_val = val_new
                H = H_new
                ac = ac_new
                max_abs_ac = max_abs_new
                s = s_new
                improved = True
            else:
                h[i] = v  # revert
    
        if not improved:
            break
    
    return h, best_val

def _generate_kaiser_init(beta):
    """Generate sign sequence from Kaiser windowed sinc."""
    n = N
    # Center
    c = (n - 1) / 2.0
    # Sinc function
    x = np.arange(n)
    sinc_vals = np.sinc(x - c) # np.sinc(x) = sin(pi*x)/(pi*x)
    # Kaiser window
    w = scipy.signal.windows.kaiser(n, beta)
    # Product
    h_cont = sinc_vals * w
    # Normalize
    h_cont = h_cont / np.max(np.abs(h_cont))
    # Sign
    h = np.sign(h_cont)
    h[h == 0] = 1.0
    return h

def _generate_blackman_harris_init():
    """Generate sign sequence from Blackman-Harris windowed sinc."""
    n = N
    c = (n - 1) / 2.0
    x = np.arange(n)
    sinc_vals = np.sinc(x - c)
    w = scipy.signal.windows.blackmanharris(n)
    h_cont = sinc_vals * w
    h_cont = h_cont / np.max(np.abs(h_cont))
    h = np.sign(h_cont)
    h[h == 0] = 1.0
    return h

def _generate_chebyshev_init(at_db):
    """Generate sign sequence from Dolph-Chebyshev windowed sinc."""
    n = N
    c = (n - 1) / 2.0
    x = np.arange(n)
    sinc_vals = np.sinc(x - c)
    w = scipy.signal.windows.chebwin(n, at_db)
    h_cont = sinc_vals * w
    h_cont = h_cont / np.max(np.abs(h_cont))
    h = np.sign(h_cont)
    h[h == 0] = 1.0
    return h

def _generate_random_smooth_init(seed):
    """Generate sign sequence from Gaussian smoothed random noise."""
    np.random.seed(seed)
    h_rand = np.random.randn(N)
    # Smooth with Gaussian
    from scipy.ndimage import gaussian_filter1d
    h_smooth = gaussian_filter1d(h_rand, sigma=5.0)
    h = np.sign(h_smooth)
    h[h == 0] = 1.0
    return h

def _generate_legendre_sequence(p, N):
    """Generate a sequence of length N based on the Legendre symbol of a prime p."""
    h = np.zeros(N)
    for i in range(N):
        a = i + 1
        if a % p == 0:
            sym = 0
        else:
            res = pow(a, (p - 1) // 2, p)
            if res == 1:
                sym = 1
            elif res == p - 1:
                sym = -1
            else:
                sym = 0
        h[i] = 1.0 if sym >= 0 else -1.0
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initialize best with the known baseline: all ones
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    # List of initializations
    inits = []
    
    # 1. Spectral designs
    # Kaiser windows with varying beta
    for beta in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]:
        inits.append(_generate_kaiser_init(beta))
        
    # Blackman-Harris
    inits.append(_generate_blackman_harris_init())
    
    # Dolph-Chebyshev
    for at_db in [20, 30, 40, 50, 60, 70, 80, 90, 100, 120]:
        inits.append(_generate_chebyshev_init(at_db))
        
    # Random smooth
    for seed in range(10):
        inits.append(_generate_random_smooth_init(seed))
        
    # Legendre sequences
    primes_to_try = [397, 401, 389, 409, 383, 419, 379, 421, 373, 431]
    for p in primes_to_try:
        inits.append(_generate_legendre_sequence(p, N))
        
    # Random +/-1
    np.random.seed(42)
    for _ in range(10):
        inits.append(np.random.choice([-1.0, 1.0], size=N))

    # Local optimization on each initialization
    for i, h_init in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Ensure sum is not zero
        s = np.sum(h_init)
        if abs(s) < N * 0.1:
            if s < 0:
                h_init = -h_init
            if abs(np.sum(h_init)) < 1e-6:
                h_init[0] = 1.0
        
        # Run local search
        # Adjust max_passes based on time remaining
        time_left = deadline - time.time()
        if time_left < 10:
            max_passes = 2
        elif time_left < 30:
            max_passes = 5
        else:
            max_passes = 10
            
        h_opt, val_opt = _local_sign_flip(h_init, max_passes=max_passes, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
            
    return best_h

# EVOLVE-BLOCK-END
