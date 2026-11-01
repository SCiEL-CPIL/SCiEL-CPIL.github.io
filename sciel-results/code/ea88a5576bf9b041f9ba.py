"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
import time
_start_time = time.time()
_DEADLINE = 150.0
_MARGIN = 5.0  # seconds safety margin

def _elapsed():
    return time.time() - _start_time

def _time_left():
    return _DEADLINE - _MARGIN - _elapsed()

def _autoconv_bound(heights):
    """
    Compute the C3 autoconvolution upper bound.
    For a step function on [-1/4, 1/4] with N equal intervals,
    the autoconvolution at the center involves sums of products.
    
    The bound C3 is related to the maximum absolute value of the
    autoconvolution normalized appropriately.
    
    For a step function h[i] on interval i, the autoconvolution at lag 0
    (center) is sum(h[i]^2) = N if |h[i]|=1.
    At other lags, it's sum(h[i]*h[i+k]).
    
    The C3 bound for the third autocorrelation inequality likely involves
    the L1 or L∞ norm of the autoconvolution sequence.
    
    We'll compute: max over all lags k of |sum_i h[i]*h[i+k]|
    normalized by N (or sqrt(N) depending on the metric).
    
    Since the paper places absolute value differently, we try the metric
    that gives the best (lowest) bound for a given function.
    """
    import numpy as np
    h = np.asarray(heights, dtype=float)
    n = len(h)
    # Autoconvolution: c[k] = sum_i h[i] * h[i+k] for valid i
    # Using FFT for efficiency
    # Zero-pad to avoid circular convolution
    size = 2 * n
    f = np.fft.rfft(h, n=size)
    ac = np.fft.irfft(f * np.conj(f), n=size)
    # ac has length size, the linear autoconvolution is in ac[0:n] and ac[size-n+1:size]
    # Actually for real input, ac[0:n] gives lags 0 to n-1, and the negative lags are symmetric
    # The full linear autoconvolution of length 2n-1:
    # ac[0:n] = c[0], c[1], ..., c[n-1]
    # ac[n+1:size] = c[-1], c[-2], ... (but reversed)
    # Let's extract properly
    c = np.zeros(2 * n - 1)
    c[0:n] = ac[0:n]
    c[n:2*n-1] = ac[n-1:0:-1]  # c[n] = c[-1], etc.
    
    # The bound is the maximum absolute value
    bound = np.max(np.abs(c))
    # Normalize: for all-ones, c[0]=N, c[k]=(N-k) for k>0, so max is N
    # Normalized by N: all-ones gives 1.0
    return bound / n

def construct():
    import numpy as np
    
    best_bound = float('inf')
    best_heights = np.ones(N)
    
    # Try several structured sign patterns
    patterns = []
    
    # 1. All ones (baseline)
    patterns.append(np.ones(N))
    
    # 2. Alternating signs
    alt = np.ones(N)
    alt[1::2] = -1
    patterns.append(alt)
    
    # 3. Block alternating (blocks of 2)
    block2 = np.ones(N)
    for i in range(0, N, 4):
        block2[i+2:i+4] = -1
    patterns.append(block2)
    
    # 4. Block alternating (blocks of 4)
    block4 = np.ones(N)
    for i in range(0, N, 8):
        block4[i+4:i+8] = -1
    patterns.append(block4)
    
    # 5. Block alternating (blocks of 10)
    block10 = np.ones(N)
    for i in range(0, N, 20):
        block10[i+10:i+20] = -1
    patterns.append(block10)
    
    # 6. Block alternating (blocks of 20)
    block20 = np.ones(N)
    for i in range(0, N, 40):
        block20[i+20:i+40] = -1
    patterns.append(block20)
    
    # 7. Block alternating (blocks of 40)
    block40 = np.ones(N)
    for i in range(0, N, 80):
        block40[i+40:i+80] = -1
    patterns.append(block40)
    
    # 8. Sine-like sign pattern: sign(sin(2*pi*x/L)) for various L
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    for L in [0.25, 0.5, 1.0, 2.0, 4.0]:
        pat = np.sign(np.sin(2 * np.pi * x / L))
        # Replace zeros with 1
        pat[pat == 0] = 1
        patterns.append(pat)
    
    # 9. Random patterns (a few)
    rng = np.random.default_rng(42)
    for _ in range(10):
        patterns.append(rng.choice([-1.0, 1.0], size=N))
    
    # 10. Pattern with more structure: sign of a quadratic
    for a in [1.0, 2.0, 4.0, 8.0]:
        pat = np.sign(a * (x**2 - 0.01))
        pat[pat == 0] = 1
        patterns.append(pat)
    
    # 11. Pattern: sign of (x - offset) for various offsets
    for offset in [-0.1, -0.05, 0.0, 0.05, 0.1]:
        pat = np.sign(x - offset)
        pat[pat == 0] = 1
        patterns.append(pat)
    
    # Evaluate all patterns
    for h in patterns:
        if _time_left() < 2.0:
            break
        bound = _autoconv_bound(h)
        if bound < best_bound:
            best_bound = bound
            best_heights = h.copy()
    
    return best_heights
