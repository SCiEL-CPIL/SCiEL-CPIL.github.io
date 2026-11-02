"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
import numpy as np


def _autoconv_peak(x):
    """Compute the maximum absolute value of the autoconvolution of x.
    Uses FFT for efficiency."""
    n = len(x)
    # Autoconvolution: c[k] = sum_j x[j] * x[j+k] for k in -(n-1)..(n-1)
    # Using FFT: ifft(fft(x, 2n) * conj(fft(x, 2n))) gives circular autoconv
    # We want linear autoconv, so zero-pad to at least 2n-1
    size = 1 << (2 * n - 1).bit_length()  # next power of 2 >= 2n-1
    X = np.fft.rfft(x, size)
    ac = np.fft.irfft(X * np.conj(X), size)
    # The linear autoconvolution is in ac[0:n] (lags 0 to n-1) and ac[size-n+1:size] (negative lags)
    # Actually, for real signals, ac is symmetric: ac[k] = ac[size-k]
    # The full autoconv has 2n-1 values. Let's extract them properly.
    # ac[0] = sum x^2 (lag 0)
    # ac[1:n] = lags 1 to n-1
    # ac[size-n+1 : size] = negative lags (by symmetry, same as positive)
    # So max abs is max of abs(ac[0:n]) since it's symmetric
    return np.max(np.abs(ac[:n]))


def _score(x):
    """Compute the C3' score: max |autoconv| / (integral)^2.
    For discrete: max |autoconv| / (sum x)^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return _autoconv_peak(x) / (s * s)


def _round_to_signed(x):
    """Round x to signed values. The task says 'signed step heights'.
    We interpret this as values in {-1, 0, 1} or perhaps just {-1, 1}.
    Given the task mentions 'signed equal-width step functions', and the seed uses np.ones(N),
    let's try rounding to nearest integer, then clip to [-1, 1], then handle zeros.
    Actually, looking at the inspiration: '±1 sign-pattern local search' scored 1.143.
    Let's round to nearest and clip to [-1,1], but ensure no zeros if possible, 
    or just use sign(). Let's try: round to nearest, clip to [-1,1].
    But wait, the task says 'signed step heights'. The seed is all 1s.
    The inspiration says '±1 sign-pattern'. Let's try rounding to {-1, 1} only.
    If x_i >= 0, set to 1, else -1.
    """
    # Try a few strategies for rounding and pick the best
    candidates = []
    
    # Strategy 1: sign(x), with 0 mapped to 1
    s1 = np.sign(x)
    s1[s1 == 0] = 1.0
    candidates.append(s1)
    
    # Strategy 2: round to nearest, clip to [-1, 1]
    s2 = np.clip(np.round(x), -1, 1)
    candidates.append(s2)
    
    # Strategy 3: threshold at 0
    s3 = np.where(x >= 0, 1.0, -1.0)
    candidates.append(s3)
    
    best = None
    best_score = np.inf
    for c in candidates:
        sc = _score(c)
        if sc < best_score:
            best_score = sc
            best = c
    return best


def _local_search(x, max_iter=2000, time_limit=10.0):
    """Local search: flip individual bits to improve score."""
    x = x.copy()
    current_score = _score(x)
    start = time.time()
    
    for i in range(max_iter):
        if time.time() - start > time_limit:
            break
        # Try flipping each element
        best_j = -1
        best_new_score = current_score
        for j in range(N):
            old_val = x[j]
            x[j] = -old_val  # flip sign
            new_score = _score(x)
            x[j] = old_val
            if new_score < best_new_score:
                best_new_score = new_score
                best_j = j
        if best_j >= 0:
            x[best_j] = -x[best_j]
            current_score = best_new_score
        else:
            break
    return x, current_score


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # Initial guess: random signs
    rng = np.random.default_rng(42)
    x = rng.choice([-1.0, 1.0], size=N)
    best_x = x.copy()
    best_score = _score(x)
    
    # Run local search
    x, sc = _local_search(x, max_iter=5000, time_limit=50.0)
    if sc < best_score:
        best_score = sc
        best_x = x.copy()
    
    # Try a few random restarts if time permits
    for restart in range(10):
        if time.time() > deadline - 30.0:
            break
        x = rng.choice([-1.0, 1.0], size=N)
        x, sc = _local_search(x, max_iter=2000, time_limit=20.0)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
    
    return best_x
# EVOLVE-BLOCK-END
