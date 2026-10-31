import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _autoconv_max_abs(x):
    """Compute max absolute value of linear autoconvolution of x."""
    n = len(x)
    # Use FFT for efficiency
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f), n=2 * n)
    # ac has length 2n, lags from -(n-1) to n-1
    # We want lags 0 to n-1 (or -n+1 to n-1), take max abs
    # ac[0] is lag 0, ac[1:n] are positive lags, ac[n+1:] are negative lags
    # Actually for linear autocorr via FFT: ac[k] for k=0..n-1 is sum x[i]*x[i+k]
    # Let's just use the full array and take max abs
    return np.max(np.abs(ac))

def _score(x):
    """Compute the score: max_abs_autoconv / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if s == 0:
        return np.inf
    return _autoconv_max_abs(x) / (s * s)

def _chirp_init(N, T):
    """Generate a signed chirp sequence based on quadratic phase."""
    t = np.arange(N)
    phase = np.pi * t * t / T
    # Map to signs: use sign of cos(phase) or threshold
    # To get ±1, use sign of real part of exp(i*phase)
    x = np.sign(np.cos(phase))
    # Ensure no zeros (sign(0) = 0)
    x[x == 0] = 1
    return x.astype(np.float64)

def _random_init(N):
    return np.random.choice([-1.0, 1.0], size=N)

def _local_search(x, max_iters=50000, deadline=None):
    """Local search: flip individual or triple elements to reduce score."""
    best = x.copy()
    best_score = _score(best)
    
    # Precompute for speed
    n = len(best)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try single flips
        improved = False
        # Random subset of positions to try
        candidates = np.random.choice(n, size=min(n, 200), replace=False)
        
        for idx in candidates:
            new_val = -best[idx]
            old_val = best[idx]
            best[idx] = new_val
            new_score = _score(best)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best[idx] = old_val
        
        if not improved:
            # Try triple flips
            triple_improved = False
            for _ in range(50):
                i, j, k = np.random.choice(n, size=3, replace=False)
                orig = best[[i, j, k]].copy()
                best[[i, j, k]] = -orig
                new_score = _score(best)
                if new_score < best_score:
                    best_score = new_score
                    triple_improved = True
                else:
                    best[[i, j, k]] = orig
            if not triple_improved and not improved:
                break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Try multiple initializations
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Chirp-based initializations with different T values
    T_values = [1.0, 2.0, 4.0, 8.0, 16.0, 3.0, 7.0, 13.0, 21.0, 50.0, 100.0, 200.0, 350.0]
    
    for T in T_values:
        if time.time() > deadline:
            break
        x = _chirp_init(N, T)
        s = _score(x)
        if s < best_score:
            best_score = s
            best_x = x.copy()
        
        # Local search from this init
        if time.time() < deadline - 10.0:
            x_ls, s_ls = _local_search(x, max_iters=20000, deadline=deadline - 5.0)
            if s_ls < best_score:
                best_score = s_ls
                best_x = x_ls.copy()
    
    # Also try random initializations
    for _ in range(5):
        if time.time() > deadline:
            break
        x = _random_init(N)
        s = _score(x)
        if s < best_score:
            best_score = s
            best_x = x.copy()
        
        if time.time() < deadline - 10.0:
            x_ls, s_ls = _local_search(x, max_iters=20000, deadline=deadline - 5.0)
            if s_ls < best_score:
                best_score = s_ls
                best_x = x_ls.copy()
    
    # Final polish on best
    if time.time() < deadline - 5.0:
        x_ls, s_ls = _local_search(best_x, max_iters=50000, deadline=deadline - 2.0)
        if s_ls < best_score:
            best_score = s_ls
            best_x = x_ls.copy()
    
    return best_x
# EVOLVE-BLOCK-END
