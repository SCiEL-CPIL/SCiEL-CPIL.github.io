"""Combined program: spectral shaping + fast local search for signed step functions."""
import numpy as np
import time
import scipy.fft as fft

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.fft as fft


def _autoconvolve(x):
    """Compute linear autoconvolution of x via FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    X = fft.rfft(x, size)
    ac = fft.irfft(X * X.conj(), size)
    return ac[:2 * n - 1]


def _objective(x):
    """Compute the ratio: max|autoconv| / (sum(x))^2. Lower is better."""
    total = np.sum(x)
    if abs(total) < 1e-12:
        return np.inf
    ac = _autoconvolve(x)
    return float(np.max(np.abs(ac))) / (total * total)


def _build_rudin_shapiro(n):
    """Construct a deterministic Rudin-Shapiro sign sequence of length n."""
    A = np.array([1.0])
    B = np.array([1.0])
    while len(A) < n:
        new_A = np.concatenate([A, B])
        new_B = np.concatenate([A, -B])
        A = new_A
        B = new_B
    return A[:n]


def _spectral_shape(rng, n, base_signal=None):
    """
    Generate a sign sequence by shaping the spectrum.
    Start with a random or base signal, modify its spectrum to reduce peak autocorrelation,
    then threshold to signs.
    """
    if base_signal is None:
        x = rng.choice([-1.0, 1.0], size=n)
    else:
        x = base_signal.copy()
    
    # Compute current spectrum
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    X = fft.rfft(x, size)
    
    # The autocorrelation is the IFFT of |X|^2.
    # To reduce peak autocorrelation, we can try to flatten |X|^2.
    # A simple approach: take the square root of |X|^2 with a phase that preserves realness.
    # Actually, let's try a different approach: 
    # We want to find signs s such that the autocorrelation of s is flat.
    # This is equivalent to finding a Barker-like sequence or a Zadoff-Chu sequence.
    
    # Let's try a heuristic: 
    # 1. Start with a random phase in the frequency domain.
    # 2. Take IFFT.
    # 3. Threshold to signs.
    # 4. This is essentially a random search in sign space guided by spectral flatness.
    
    # Alternative: Use the fact that for a ±1 sequence, the autocorrelation at lag k
    # is related to the number of agreements minus disagreements.
    
    # Let's try a simple spectral shaping:
    # Reduce the magnitude of high-frequency components to smooth the autocorrelation.
    # Actually, we want to REDUCE the PEAK of the autocorrelation, which is at lag 0.
    # Lag 0 autocorrelation is always sum(x^2) = n for ±1 sequences.
    # So we can't reduce the lag-0 value. We need to reduce other lags or make them negative.
    # The max |autocorr| will be at least n (at lag 0).
    # Wait, the objective is max |autocorr(k)| / (sum x)^2.
    # For ±1 sequences, sum(x) can be small, making the denominator small.
    # But if sum(x) is close to 0, the denominator is close to 0, making the ratio large.
    # So we want sum(x) to be large AND max |autocorr| to be small.
    # For all-ones, sum = n, max |autocorr| = n, ratio = n/n^2 = 1/n.
    # For a balanced ±1 sequence, sum ≈ 0, so ratio is large.
    # So the all-ones sequence is actually quite good because the denominator is large.
    
    # Let's try sequences that are mostly +1 with a few -1s.
    # Or sequences with a slow variation.
    
    # Let's try a different approach: 
    # Start with all ones, and flip a few signs to reduce the autocorrelation peak.
    # The autocorrelation at lag k for all-ones is n - |k|.
    # If we flip one sign at position i, the autocorrelation changes.
    
    # For now, let's just return the input or a random sequence.
    return x


def _hill_climb_fast(x, max_iter=500, deadline=140.0, start_time=0.0):
    """
    Local search: try flipping individual signs to reduce max |autoconvolution| / (sum)^2.
    Use FFT-based autoconvolution for speed.
    """
    current_val = _objective(x)
    if np.isinf(current_val):
        return x, current_val
    
    # Precompute autoconvolution
    c = _autoconvolve(x)
    current_max = float(np.max(np.abs(c)))
    current_sum = float(np.sum(x))
    current_sum_sq = current_sum * current_sum
    current_val = current_max / current_sum_sq if current_sum_sq > 0 else np.inf
    
    # Track best
    best_x = x.copy()
    best_val = current_val
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random index to try flipping
        idx = np.random.randint(N)
        
        # Skip if flipping would make sum closer to 0 (bad for denominator)
        # Actually, let's just try it and see.
        
        x_copy = x.copy()
        x_copy[idx] = -x_copy[idx]
        
        # Quick check: if sum becomes 0, skip
        new_sum = current_sum - 2 * x[idx]
        if abs(new_sum) < 1e-12:
            continue
        
        # Compute new autoconvolution
        c_new = _autoconvolve(x_copy)
        new_max = float(np.max(np.abs(c_new)))
        new_sum_sq = new_sum * new_sum
        new_val = new_max / new_sum_sq
        
        if new_val < current_val:
            x = x_copy
            c = c_new
            current_max = new_max
            current_sum = new_sum
            current_sum_sq = new_sum_sq
            current_val = new_val
            
            if current_val < best_val:
                best_val = current_val
                best_x = x.copy()
    
    return best_x, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_h = None
    best_val = float('inf')
    
    # Strategy 1: All ones (baseline, score 1.0)
    h_ones = np.ones(N)
    val_ones = _objective(h_ones)
    best_h = h_ones.copy()
    best_val = val_ones
    
    # Strategy 2: Rudin-Shapiro
    h_rs = _build_rudin_shapiro(N)
    val_rs = _objective(h_rs)
    if val_rs < best_val:
        best_val = val_rs
        best_h = h_rs.copy()
    
    # Strategy 3: Random starts + local search
    for seed in range(30):
        if time.time() > deadline:
            break
        
        rng = np.random.default_rng(seed)
        
        # Start with a sequence that has a large sum (mostly +1s)
        # Flip a small fraction of signs
        frac_flip = rng.uniform(0.01, 0.1)
        n_flip = int(frac_flip * N)
        h_rand = np.ones(N)
        flip_idx = rng.choice(N, size=n_flip, replace=False)
        h_rand[flip_idx] = -1.0
        
        # Quick evaluation
        curr_val = _objective(h_rand)
        if np.isinf(curr_val):
            continue
        
        # Local search
        h_improved, improved_val = _hill_climb_fast(h_rand, max_iter=200, deadline=deadline, start_time=start_time)
        
        if improved_val < best_val:
            best_val = improved_val
            best_h = h_improved.copy()
    
    # Strategy 4: Local search from Rudin-Shapiro
    if time.time() < deadline:
        h_rs2 = best_h.copy() if best_h is not None else np.ones(N)
        h_opt, opt_val = _hill_climb_fast(h_rs2, max_iter=500, deadline=deadline, start_time=start_time)
        if opt_val < best_val:
            best_val = opt_val
            best_h = h_opt.copy()
    
    # Final fallback: if something went wrong, return all ones
    if best_h is None:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
