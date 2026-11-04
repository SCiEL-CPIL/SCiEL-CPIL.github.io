"""Optimized PSL minimization for signed step function with N=400 intervals."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _score(h):
    """Compute the upper bound on C3: max absolute autocorrelation at non-zero lags."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))


def _compute_acf(h):
    """Compute full autocorrelation via FFT."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return acf


def _psl_value(acf, n):
    """Get PSL value from acf array (max abs at non-zero lags)."""
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))


def _gradient_step(h, acf):
    """
    Gordon's fast PSL minimization step.
    For each position i, compute the change in PSL if we flip h[i].
    Returns (best_i, new_psl, current_psl).
    """
    n = len(h)
    current_psl = _psl_value(acf, n)
    
    # Build S matrix: S[i,k] = h[i-k] + h[i+k] with boundary handling
    S = np.zeros((n, n - 1), dtype=np.float64)
    for k in range(1, n):
        idx = k - 1
        # h[i-k] for i = k, k+1, ..., n-1
        S[k:, idx] += h[:n - k]
        # h[i+k] for i = 0, 1, ..., n-1-k
        S[:n - k, idx] += h[k:]
    
    # delta[i, k] = -2 * h[i] * S[i, k]
    delta = -2.0 * h[:, np.newaxis] * S
    
    acf_lags = acf[1:n]  # shape (n-1,)
    new_acf = delta + acf_lags[np.newaxis, :]
    new_psl = np.max(np.abs(new_acf), axis=1)
    
    best_i = int(np.argmin(new_psl))
    best_new_psl = new_psl[best_i]
    
    return best_i, float(best_new_psl), current_psl


def _fast_psl_optimize(h, max_passes=500, time_budget=20.0, start_time=None):
    """
    Fast PSL optimization using Gordon's gradient method.
    Repeatedly find the single best flip and apply it.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    acf = _compute_acf(h)
    best_psl = _psl_value(acf, n)
    
    passes = 0
    while passes < max_passes:
        if time.time() - start_time > time_budget:
            break
        
        i, new_psl, cur_psl = _gradient_step(h, acf)
        
        if new_psl >= cur_psl:
            break
        
        h[i] = -h[i]
        passes += 1
        
        acf = _compute_acf(h)
        cur_psl = _psl_value(acf, n)
        
        if cur_psl < best_psl:
            best_psl = cur_psl
            best_h = h.copy()
        
        if best_psl < 1e-10:
            break
    
    return best_h, best_psl


def _simulated_annealing_psl(h, time_budget=15.0, start_time=None):
    """
    Simulated annealing on PSL.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    acf = _compute_acf(h)
    current_psl = _psl_value(acf, n)
    
    best_h = h.copy()
    best_psl = current_psl
    
    T = 5.0
    T_min = 0.1
    cooling = 0.97
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        i = np.random.randint(n)
        
        # Compute delta for this flip
        delta = np.zeros(n - 1)
        for k in range(1, n):
            contrib = 0.0
            if i - k >= 0:
                contrib += h[i - k]
            if i + k < n:
                contrib += h[i + k]
            delta[k - 1] = -2.0 * h[i] * contrib
        
        new_acf_lags = acf[1:n] + delta
        new_psl = float(np.max(np.abs(new_acf_lags)))
        
        delta_psl = new_psl - current_psl
        
        if delta_psl < 0 or np.random.random() < np.exp(-delta_psl / T):
            h[i] = -h[i]
            acf = _compute_acf(h)
            current_psl = _psl_value(acf, n)
            
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
        
        T *= cooling
    
    return best_h, best_psl


def _sparse_init(n, density=0.5, seed=0):
    """
    Initialize a sparse-ish sequence: pick a subset of positions to be non-zero
    (with random signs), rest are zero. But wait - the task says "signed step function"
    which likely means values are from {-1, 1}. Let me re-read...
    
    Actually the task says "signed step function with 400 equal intervals". This typically
    means the function takes values in {-1, 1} on each interval. So sparse init with zeros
    might not be valid. Let me just use random ±1 init but with structure.
    
    I'll use a different strategy: initialize with a pattern based on a low-discrepancy
    sequence or a structured pattern that has good autocorrelation properties initially.
    """
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=n)
    return h


def _structured_init(n, seed=0):
    """
    Create structured initial sequences with better autocorrelation properties.
    """
    rng = np.random.default_rng(seed)
    
    # Option 1: Random
    h1 = rng.choice([-1.0, 1.0], size=n)
    
    # Option 2: Based on quadratic residue (Barker-like)
    h2 = np.array([1.0 if (i * i) % 7 in [0, 1, 2, 4] else -1.0 for i in range(n)])
    
    # Option 3: Alternating with random perturbations
    h3 = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
    flip_mask = rng.random(n) < 0.1
    h3[flip_mask] *= -1
    
    # Option 4: Based on sine wave
    h4 = np.sign(np.sin(2 * np.pi * rng.random() * i / n + rng.random() * 2 * np.pi))
    h4 = np.where(h4 == 0, 1.0, h4)
    
    return [h1, h2, h3, h4]


def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound (max absolute autocorrelation at non-zero lags).
    
    Strategy:
    1. Try multiple structured and random initializations.
    2. For each, run fast gradient-based PSL optimization (Gordon's method).
    3. Use simulated annealing to escape local minima.
    4. Return the best sequence found.
    """
    start_time = time.time()
    deadline = 135.0  # Safety margin: 150s limit, leave 15s
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Fast gradient optimization from multiple diverse starts
    n_starts = 15
    for s in range(n_starts):
        if time.time() > deadline - 30:  # Leave time for SA
            break
        
        # Get initial candidates
        candidates = _structured_init(N, seed=s * 7919 + 13)
        
        for h_init in candidates:
            if time.time() > deadline - 30:
                break
            
            # Run fast gradient optimization
            h_opt, psl_opt = _fast_psl_optimize(
                h_init, 
                max_passes=300,
                time_budget=10.0,
                start_time=start_time
            )
            
            if psl_opt < best_score:
                best_score = psl_opt
                best_h = h_opt.copy()
    
    # Phase 2: Simulated annealing from the best found so far
    if best_h is not None:
        if time.time() < deadline - 5:
            h_sa, psl_sa = _simulated_annealing_psl(
                best_h,
                time_budget=min(15.0, deadline - 5 - (time.time() - start_time)),
                start_time=start_time
            )
            
            if psl_sa < best_score:
                best_score = psl_sa
                best_h = h_sa
    
    # Phase 3: Final polish with gradient steps if time permits
    if best_h is not None and time.time() < deadline - 2:
        remaining_time = deadline - 2 - (time.time() - start_time)
        if remaining_time > 1:
            h_final, psl_final = _fast_psl_optimize(
                best_h,
                max_passes=100,
                time_budget=remaining_time,
                start_time=start_time
            )
            if psl_final < best_score:
                best_score = psl_final
                best_h = h_final
    
    # Ensure we have a valid result
    if best_h is None:
        best_h = np.random.choice([-1.0, 1.0], size=N)
        best_score = _score(best_h)
    
    return best_h

# EVOLVE-BLOCK-END
