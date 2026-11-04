"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_peak(h):
    """Compute max |autocorrelation| at non-zero lags using FFT."""
    n = len(h)
    if n <= 1:
        return 0.0
    # Zero-pad to 2*n for linear autocorrelation
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # acf[0] is lag 0 (sum of squares), we want max |acf[k]| for k=1..n-1
    # Also consider negative lags: acf[n+1] to acf[fft_size-1] correspond to negative lags
    # But for real sequences, acf is symmetric, so max over 1..n-1 suffices
    return float(np.max(np.abs(acf[1:n])))


def _phase_based_init(num_harmonics=8, seed=42):
    """Generate initial ±1 sequence using quasi-periodic phase alignment.
    
    Uses a sum of complex exponentials with irrational frequency ratios to create
    a structured sequence that inherently has low autocorrelation peaks.
    """
    rng = np.random.RandomState(seed)
    n = N
    
    # Generate irrational frequency ratios using golden ratio conjugate
    golden = (1 + np.sqrt(5)) / 2.0
    # Frequencies are k * golden for k=1..num_harmonics, normalized
    freqs = np.array([(k * golden) % 1.0 for k in range(1, num_harmonics + 1)])
    
    # Random phases for each harmonic
    phases = 2 * np.pi * rng.rand(num_harmonics)
    
    # Sample the sum of complex exponentials at n points
    t = np.linspace(0, 1, n, endpoint=False)
    signal = np.zeros(n, dtype=np.complex128)
    for k in range(num_harmonics):
        signal += np.exp(2j * np.pi * freqs[k] * t + phases[k])
    
    # Take real part and threshold to ±1
    real_signal = np.real(signal)
    # Normalize to have zero mean for better balance
    real_signal = real_signal - np.mean(real_signal)
    
    # Threshold: positive → +1, negative → -1
    h = np.sign(real_signal)
    # Handle zeros by assigning +1
    h[h == 0] = 1.0
    
    return h


def _local_search(h, max_time):
    """Greedy local search: flip bits to reduce autocorrelation peak."""
    n = len(h)
    h = h.copy()
    current_peak = _autocorr_peak(h)
    start_time = time.time()
    
    # Precompute positions
    positions = np.arange(n)
    
    while time.time() - start_time < max_time:
        improved = False
        # Shuffle positions to avoid systematic bias
        np.random.shuffle(positions)
        
        for i in positions:
            if time.time() - start_time > max_time:
                break
            # Try flipping position i
            old_val = h[i]
            h[i] = -h[i]
            new_peak = _autocorr_peak(h)
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
            else:
                h[i] = old_val
        
        if not improved:
            # Try flipping pairs of positions
            for _ in range(100):
                if time.time() - start_time > max_time:
                    break
                i, j = np.random.choice(n, 2, replace=False)
                h[i] = -h[i]
                h[j] = -h[j]
                new_peak = _autocorr_peak(h)
                if new_peak < current_peak:
                    current_peak = new_peak
                    improved = True
                else:
                    h[i] = -h[i]
                    h[j] = -h[j]
            
            if not improved:
                break
    
    return h, current_peak


def construct():
    """Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound using phase-based initialization."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Try multiple phase-based initializations with different parameters
    best_h = None
    best_peak = float('inf')
    
    # Try different numbers of harmonics and seeds
    configs = [
        (4, 42), (6, 42), (8, 42), (10, 42),
        (6, 123), (8, 123), (10, 123),
        (4, 456), (6, 456), (8, 456),
    ]
    
    for num_harm, seed in configs:
        if time.time() > deadline - 5:
            break
        
        # Generate phase-based initial sequence
        h_init = _phase_based_init(num_harm, seed)
        
        # Run local search with remaining time budget
        remaining = deadline - time.time() - 5
        if remaining < 2:
            break
        
        h_candidate, peak_candidate = _local_search(h_init, remaining)
        
        if peak_candidate < best_peak:
            best_peak = peak_candidate
            best_h = h_candidate
    
    # If no good result, fall back to random initialization
    if best_h is None:
        np.random.seed(42)
        best_h = np.random.choice([-1.0, 1.0], size=N)
    
    return best_h
# EVOLVE-BLOCK-END
