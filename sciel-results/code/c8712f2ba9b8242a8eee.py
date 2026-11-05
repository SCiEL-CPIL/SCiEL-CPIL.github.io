import numpy as np
import time

N = 400

def _score(h):
    """Compute the max absolute autocorrelation for lags 1 to N-1."""
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return np.max(np.abs(acf[1:N]))

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

def _compute_neighbor_sums(h):
    """
    Compute for each position i the vector of neighbor sums:
    neighbor_sum[i][k-1] = h[i-k] + h[i+k] for valid k.
    Returns array of shape (n, n-1).
    """
    n = len(h)
    neighbor_sums = np.zeros((n, n-1))
    for k in range(1, n):
        neighbor_sums[k:, k-1] += h[:n-k]
        neighbor_sums[:n-k, k-1] += h[k:]
    return neighbor_sums

def _fast_psl_optimize(h, max_passes=500, time_budget=None, start_time=None):
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
    neighbor_sums = _compute_neighbor_sums(h)
    
    passes = 0
    while passes < max_passes:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        # Vectorized best flip search
        deltas = -2.0 * h[:, np.newaxis] * neighbor_sums
        new_acf_lags_all = acf[1:n][np.newaxis, :] + deltas
        new_psls = np.max(np.abs(new_acf_lags_all), axis=1)
        
        best_i = int(np.argmin(new_psls))
        best_new_psl = float(new_psls[best_i])
        
        if best_new_psl >= best_psl:
            break
        
        # Flip h[best_i]
        old_val = h[best_i]
        h[best_i] = -old_val
        passes += 1
        
        # Update neighbor sums incrementally
        # When h[i] flips from old to -old, change is -2*old
        # For each lag k:
        #   neighbor_sums[i+k, k-1] and neighbor_sums[i-k, k-1] are affected
        for k in range(1, n):
            if i + k < n:
                neighbor_sums[i+k, k-1] -= 2.0 * old_val
            if i - k >= 0:
                neighbor_sums[i-k, k-1] -= 2.0 * old_val
        
        # Recompute acf
        acf = _compute_acf(h)
        cur_psl = _psl_value(acf, n)
        
        if cur_psl < best_psl:
            best_psl = cur_psl
            best_h = h.copy()
        
        if best_psl < 1e-10:
            break
    
    return best_h, best_psl

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/- 1."""
    t = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        t[i] = t[i >> 1] ^ (i & 1)
    return 1.0 - 2.0 * t

def _chaotic_sequence(n, seed):
    """Generate a chaotic sequence using logistic map."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 0.9)
    r = 4.0  # Fully chaotic
    signs = np.zeros(n)
    for i in range(n):
        signs[i] = 1.0 if x > 0.5 else -1.0
        x = r * x * (1.0 - x)
        if x <= 0 or x >= 1:
            x = rng.uniform(0.1, 0.9)
    return signs

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Thue-Morse (known to have good autocorrelation properties)
    candidates.append(_thue_morse(N))
    
    # 2. Random sequences
    for i in range(15):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # 3. Chaotic sequences
    for seed in range(10):
        candidates.append(_chaotic_sequence(N, seed))
    
    # 4. Quasi-periodic sequences
    for trial in range(5):
        phases = rng.uniform(0, 2*np.pi, size=5)
        amps = rng.uniform(0.5, 1.0, size=5)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for k in range(1, 6):
            val += amps[k-1] * np.cos(2*np.pi*k*t + phases[k-1])
        candidates.append(np.sign(val))
    
    # 5. Block patterns
    for block_size in [5, 10, 20]:
        for trial in range(3):
            h = np.zeros(N)
            for b in range(N // block_size):
                h[b*block_size:(b+1)*block_size] = rng.choice([-1.0, 1.0])
            candidates.append(h)
    
    # Evaluate all candidates
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine top candidates
    for i, (s, h) in enumerate(scored_candidates[:10]):
        if time.time() > deadline - 10:
            break
        time_limit = deadline - 5
        h_refined, s_refined = _fast_psl_optimize(h, max_passes=1000, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # Random restarts with optimization
    restart_count = 0
    while time.time() < deadline - 5 and restart_count < 30:
        restart_count += 1
        h_init = rng.choice([-1.0, 1.0], size=N)
        time_limit = min(deadline - 3, time.time() + 2)
        h_refined, s_refined = _fast_psl_optimize(h_init, max_passes=500, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
