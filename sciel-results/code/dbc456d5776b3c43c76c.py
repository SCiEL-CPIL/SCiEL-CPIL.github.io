"""Additive bases with few representations - FFT-accelerated simulated annealing.

We model the problem as finding a subset A of [0, 128] such that every integer
in [128, 256] is a sum of two elements of A. We minimize the largest representation
multiplicity (peak).

Key structural change from the seed: use the Fast Fourier Transform (FFT) to compute
the autocorrelation (ordered pair counts) of the indicator vector of A in O(N log N)
instead of O(N^2) convolution. This allows many more simulated annealing steps within
the time budget, improving the chance of escaping local optima.

The unordered pair count for sum k is:
  U_k = (ordered_count[k] + diag[k]) / 2
where ordered_count[k] = sum_{i+j=k} x_i * x_j
and diag[k] = 1 if k is even and x_{k/2} = 1, else 0.

We use a variance-aware objective to encourage flatter distributions, which often
leads to lower peaks.
"""
import time
import random
import numpy as np

_N = 128
_LEN = 129  # indices 0..128
_T0 = 128
_T1 = 256
_WIN = _T1 - _T0 + 1  # 129 targets

_START = time.time()
_DEADLINE = 85.0  # safety margin before the 90s kill


def _elapsed():
    return time.time() - _START


def _fft_ordered_counts(x):
    """Compute ordered pair counts for targets [128, 256] using FFT.
    
    ordered_count[k] = sum_{i+j=k} x_i * x_j
    This is the auto-correlation of x, which can be computed via FFT:
      FFT(x) * conj(FFT(x)) -> IFFT -> ordered_count
    """
    # Pad to at least 2*_LEN - 1 to avoid circular convolution
    n_fft = 1 << (2 * _LEN - 1).bit_length()  # next power of 2 >= 2*_LEN-1
    X = np.fft.rfft(x.astype(np.float64), n=n_fft)
    # Auto-correlation: IFFT(FFT(x) * conj(FFT(x)))
    corr = np.fft.irfft(X * np.conj(X), n=n_fft)
    # corr[k] = sum_{i} x[i] * x[i+k] for k >= 0, but we need sum_{i+j=k} x[i]*x[j]
    # Actually, np.fft.irfft(X * np.conj(X)) gives the circular cross-correlation.
    # For linear auto-correlation, we need to be careful.
    # Let's use: ordered_count[k] = sum_{i=0}^{k} x[i] * x[k-i]
    # This is the convolution of x with x (reversed), but since x is the indicator,
    # convolution of x with x gives sum_{i+j=k} x[i]*x[j].
    # So we can just use np.convolve or FFT-based convolution.
    # Let me use FFT-based convolution:
    Y = np.fft.rfft(x.astype(np.float64), n=n_fft)
    conv = np.fft.irfft(Y * Y, n=n_fft)
    # conv[k] = sum_{i+j=k} x[i]*x[j] for k in [0, 2*_LEN-2]
    return conv[_T0:_T1 + 1].astype(np.int64)


def _compute_counts(x):
    """Compute unordered pair counts for targets [128, 256]."""
    ordered = _fft_ordered_counts(x)
    ks = np.arange(_T0, _T1 + 1)
    is_even = (ks % 2 == 0)
    mids = ks // 2
    diag_corr = np.where(is_even, x[mids], 0)
    u_counts = (ordered + diag_corr) // 2
    return u_counts


def _valid(u_counts):
    return bool(np.all(u_counts > 0))


def _peak(u_counts):
    return int(u_counts.max())


def _variance(u_counts):
    return float(np.var(u_counts))


def _make_greedy_seed(rng):
    """Build a valid seed using greedy coverage with load balancing."""
    x = np.zeros(_LEN, dtype=np.int64)
    u = _compute_counts(x)
    
    candidates = np.arange(_LEN)
    rng.shuffle(candidates)
    
    # First pass: ensure coverage
    while not _valid(u):
        uncovered = np.where(u == 0)[0] + _T0
        if len(uncovered) == 0:
            break
        
        best_c = None
        best_score = -1e18
        best_new_u = None
        
        for c in candidates:
            if x[c] == 1:
                continue
            
            # Compute new counts if we add c
            # Delta: for each target k in [128, 256], if k-c is in A, add 1.
            # Also if k == 2c, add 1 (diagonal).
            k_min = max(_T0, c)
            k_max = min(_T1, c + _N)
            
            if k_min > k_max:
                continue
            
            ks = np.arange(k_min, k_max + 1)
            partners = ks - c
            # Valid partners are in [0, 128]
            valid_mask = (partners >= 0) & (partners < _LEN)
            
            new_u = u.copy()
            # Add contributions from existing elements
            mask = valid_mask & (x[partners] == 1)
            idxs = ks - _T0
            new_u[idxs[mask]] += 1
            
            # Handle diagonal (2c == k)
            if _T0 <= 2 * c <= _T1:
                new_u[2 * c - _T0] += 1
            
            if np.all(new_u > 0):
                peak_new = new_u.max()
                var_new = np.var(new_u)
                # Score: minimize peak, then variance
                score = -peak_new - 0.01 * var_new
                if score > best_score:
                    best_score = score
                    best_c = c
                    best_new_u = new_u
        
        if best_c is None:
            # Fallback: pick any candidate that covers *any* uncovered target
            for c in candidates:
                if x[c] == 1:
                    continue
                covered_any = False
                for t in uncovered:
                    p = t - c
                    if (0 <= p < _LEN and x[p] == 1) or (2 * c == t):
                        covered_any = True
                        break
                if covered_any:
                    best_c = c
                    break
        
        if best_c is None:
            best_c = candidates[0]
        
        x[best_c] = 1
        if best_new_u is not None and best_c is not None:
            u = best_new_u
        else:
            u = _compute_counts(x)
    
    return x


def _simulated_annealing(x, rng, max_time):
    """Simulated annealing with FFT-accelerated count computation."""
    u = _compute_counts(x)
    if not _valid(u):
        return x
    
    best_x = x.copy()
    best_u = u.copy()
    best_peak = _peak(u)
    best_var = _variance(u)
    
    current_peak = best_peak
    current_var = best_var
    
    T = 10.0
    cooling = 0.9995  # Slower cooling for more exploration
    
    # Pre-compute indices for potential speedup
    while _elapsed() < max_time:
        T *= cooling
        if T < 1e-5:
            break
        
        # Random move: pick a random element to toggle
        i = rng.randint(0, _LEN - 1)
        
        x_new = x.copy()
        x_new[i] = 1 - x_new[i]
        
        u_new = _compute_counts(x_new)
        
        if not _valid(u_new):
            # Reject invalid moves
            continue
        
        peak_new = _peak(u_new)
        var_new = _variance(u_new)
        
        # Objective: minimize peak, then variance
        obj_cur = current_peak + 0.01 * current_var
        obj_new = peak_new + 0.01 * var_new
        
        delta = obj_new - obj_cur
        
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            x = x_new
            u = u_new
            current_peak = peak_new
            current_var = var_new
            
            if peak_new < best_peak or (peak_new == best_peak and var_new < best_var):
                best_peak = peak_new
                best_var = var_new
                best_x = x.copy()
                best_u = u.copy()
                
    return best_x


def construct():
    global _START
    _START = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Get a valid seed
    x = _make_greedy_seed(rng)
    u = _compute_counts(x)
    if not _valid(u):
        # Fallback: full set
        x = np.ones(_LEN, dtype=np.int64)
        u = _compute_counts(x)
    
    # Phase 2: Simulated Annealing
    remaining = _DEADLINE - _elapsed() - 2
    if remaining > 1:
        x_opt = _simulated_annealing(x, rng, max_time=remaining)
        u_opt = _compute_counts(x_opt)
        if _valid(u_opt):
            x = x_opt
            u = u_opt
    
    # Convert to sorted list
    result = [int(i) for i in np.where(x == 1)[0]]
    return result
