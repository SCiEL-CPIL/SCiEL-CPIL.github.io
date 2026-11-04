"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400
DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _autocorr_peak(x):
    """
    Compute the peak of the autoconvolution of x.
    x is a length-N array of signed step heights on [-1/4, 1/4] with 400 equal intervals.
    The autoconvolution (circular, since we're on a periodic domain implicitly by FFT)
    peak is what we want to minimize (upper bound on C3).
    
    Actually, let's think about this more carefully.
    The task says "improve an autoconvolution upper bound" and "score is the upper bound on C3".
    Lower is better.
    
    For a step function with heights x[0..N-1] on N equal intervals,
    the autoconvolution at lag k is sum_i x[i] * x[i+k] (with appropriate wrapping or boundary handling).
    
    Since we're on [-1/4, 1/4] with 400 equal intervals, and the function is a step function,
    the continuous autoconvolution peak is what matters.
    
    Let me use circular convolution via FFT for efficiency, then find the peak (max absolute value
    or just max value? The task says "absolute value differently" in paper vs notebook).
    
    The task says: "Paper and published result notebook place the absolute value differently; execution must explicitly select a metric."
    
    I'll use the max of the absolute value of the autoconvolution as the metric to minimize.
    
    Actually, for the "third autocorrelation inequality", C3 is likely related to the L-infinity norm
    of the autocorrelation. Let me compute the circular autocorrelation and take its max absolute value.
    """
    # Use FFT for circular autocorrelation
    n = len(x)
    # Pad to avoid circular effects? Or use circular directly?
    # For a step function on a bounded interval, the natural convolution is linear, not circular.
    # But with 400 intervals, linear convolution would have length 2N-1.
    # Let me use linear convolution to be safe.
    
    # Linear autocorrelation: r[k] = sum_{i} x[i] * x[i+k] for valid k
    # Using FFT: zero-pad to size >= 2N
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    X = np.fft.fft(x, nfft)
    ac = np.fft.ifft(X * np.conj(X)).real
    # The autocorrelation is ac[0..n-1] for lags 0 to n-1, and ac[nfft-n+1..nfft-1] for negative lags
    # Peak is max absolute value over all lags
    return np.max(np.abs(ac))


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n, mapped to {-1, 1}."""
    tm = np.zeros(n, dtype=np.int8)
    for i in range(n):
        tm[i] = bin(i).count('1') % 2
    return 1 - 2 * tm  # map 0->1, 1->-1


def _local_search(x, max_iters=50000, deadline=None):
    """
    Greedy local search: try flipping individual elements and small neighborhoods.
    """
    x = x.copy()
    current_score = _autocorr_peak(x)
    best_x = x.copy()
    best_score = current_score
    
    n = len(x)
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try flipping each element
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            x[i] = -x[i]
            new_score = _autocorr_peak(x)
            if new_score < current_score - 1e-12:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
                improved = True
            else:
                x[i] = -x[i]
        iters += 1
    
    return best_x, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_x = None
    best_score = float('inf')
    
    # Try multiple initializations
    initializations = []
    
    # Thue-Morse
    initializations.append(_thue_morse(N))
    
    # Random
    rng = np.random.default_rng(42)
    for seed in range(20):
        initializations.append(rng.choice([-1.0, 1.0], size=N))
    
    # Quasi-periodic
    for freq in [3, 5, 7, 11, 13]:
        t = np.linspace(0, 1, N, endpoint=False)
        initializations.append(np.sign(np.sin(2 * np.pi * freq * t)))
    
    # Alternating
    initializations.append(np.tile([-1.0, 1.0], N // 2))
    
    # All ones (baseline)
    initializations.append(np.ones(N))
    
    # Run local search on each initialization
    for init in initializations:
        if time.time() > deadline - 5:
            break
        
        x, score = _local_search(init, max_iters=10000, deadline=deadline - 5)
        if score < best_score:
            best_score = score
            best_x = x
    
    # Final polish with more iterations if time remains
    if best_x is not None and time.time() < deadline - 2:
        x, score = _local_search(best_x, max_iters=100000, deadline=deadline - 1)
        if score < best_score:
            best_score = score
            best_x = x
    
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
