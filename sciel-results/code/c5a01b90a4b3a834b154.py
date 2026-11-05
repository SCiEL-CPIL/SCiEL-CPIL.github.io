import numpy as np
import time

N = 400

def _score(h):
    """Compute the max absolute autocorrelation for lags 1 to N-1."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
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
    Returns (best_i, best_new_psl, current_psl).
    """
    n = len(h)
    current_psl = _psl_value(acf, n)
    
    best_i = -1
    best_new_psl = float('inf')
    
    # Pre-compute the current ACF lags
    acf_lags = acf[1:n]
    
    # For each position i, the delta in ACF lags when flipping h[i] is:
    # delta[k-1] = -2 * h[i] * (h[i-k] + h[i+k]) for valid indices
    # We can precompute a "neighbor sum" matrix more efficiently.
    
    # Vectorized approach: for each i, compute neighbor_sum efficiently
    # neighbor_sum[i, k-1] = h[i-k] + h[i+k] (where valid)
    
    # Use convolution-like approach:
    # For left neighbors: convolve h with reversed h
    # For right neighbors: convolve h with h
    
    # More directly: build the delta for all i at once
    # delta[i, j] = -2*h[i] * (h[i-(j+1)] + h[i+(j+1)]) for j=0..n-2
    
    # Let's compute this efficiently using shifted arrays
    best_i = -1
    best_new_psl = float('inf')
    
    # Process in batches for memory efficiency
    batch_size = 50
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        for i in range(start, end):
            neighbor_sum = np.zeros(n-1)
            
            max_k1 = min(i, n-1)
            if max_k1 > 0:
                neighbor_sum[:max_k1] += h[i-1 : i-max_k1-1 : -1]
                
            max_k2 = min(n-1-i, n-1)
            if max_k2 > 0:
                neighbor_sum[:max_k2] += h[i+1 : i+max_k2+1]
                
            delta = -2.0 * h[i] * neighbor_sum
            new_acf_lags = acf_lags + delta
            new_psl = float(np.max(np.abs(new_acf_lags)))
            
            if new_psl < best_new_psl:
                best_new_psl = new_psl
                best_i = i
                
    return best_i, float(best_new_psl), current_psl

def _update_acf_flip(h, acf, i):
    """
    Incrementally update ACF after flipping h[i].
    delta[k] = -2 * h[i] * (h[i-k] + h[i+k]) for lag k
    """
    n = len(h)
    # Flipping h[i] changes h[i] to -h[i]
    # The ACF at lag k: sum_j h[j]*h[j+k]
    # Change: -2*h[i]*h[i-k] - 2*h[i]*h[i+k] (with appropriate bounds)
    # Actually when we flip h[i], the new h[i] = -h[i]
    # New ACF[k] = old ACF[k] - 2*h[i]*(h[i-k] + h[i+k]) for valid indices
    
    delta = np.zeros(2*n - 1)
    
    # For lag k (can be negative or positive in full ACF)
    # ACF[k] = sum_j h[j]*h[j+k]
    # When h[i] flips, terms involving h[i] change:
    # h[i]*h[i+k] -> -h[i]*h[i+k], change = -2*h[i]*h[i+k]
    # h[i-k]*h[i] -> h[i-k]*(-h[i]), change = -2*h[i]*h[i-k]
    
    for k in range(-(n-1), n):
        # Term h[i]*h[i+k] if i+k in [0,n)
        if 0 <= i+k < n:
            delta[k + n - 1] -= 2.0 * h[i] * h[i+k]
        # Term h[i-k]*h[i] if i-k in [0,n)
        if 0 <= i-k < n:
            delta[k + n - 1] -= 2.0 * h[i] * h[i-k]
    
    acf_new = acf + delta
    return acf_new

def _fast_psl_optimize(h, max_passes=200, time_budget=None, start_time=None):
    """
    Fast PSL optimization using Gordon's gradient method.
    Repeatedly find the single best flip and apply it.
    Uses incremental ACF updates.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    acf = _compute_acf(h)
    best_psl = _psl_value(acf, n)
    
    passes = 0
    no_improve_count = 0
    while passes < max_passes:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        i, new_psl, cur_psl = _gradient_step(h, acf)
        
        if i == -1 or new_psl >= cur_psl:
            no_improve_count += 1
            if no_improve_count >= 3:
                break
            continue
        
        h[i] = -h[i]
        passes += 1
        no_improve_count = 0
        
        # Incremental ACF update
        acf = _update_acf_flip(h, acf, i)
        cur_psl = _psl_value(acf, n)
        
        if cur_psl < best_psl:
            best_psl = cur_psl
            best_h = h.copy()
        
        if best_psl < 1e-10:
            break
    
    return best_h, best_psl

def _pairwise_flip_search(h, acf, max_pairs=200, time_budget=None, start_time=None):
    """
    Try flipping pairs of positions to escape local minima.
    Returns (best_h, best_psl).
    """
    n = len(h)
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    best_psl = _psl_value(acf, n)
    
    # Sample random pairs
    rng = np.random.default_rng(12345)
    attempts = 0
    while attempts < max_pairs:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        # Pick two random distinct positions
        i, j = rng.choice(n, size=2, replace=False)
        if i == j:
            attempts += 1
            continue
        
        # Try flipping both i and j
        h[i] = -h[i]
        h[j] = -h[j]
        
        # Compute new ACF (full recompute for simplicity, or incremental)
        acf_new = _compute_acf(h)
        psl_new = _psl_value(acf_new, n)
        
        if psl_new < best_psl:
            best_psl = psl_new
            best_h = h.copy()
            acf = acf_new
        else:
            # Revert
            h[i] = -h[i]
            h[j] = -h[j]
        
        attempts += 1
    
    return best_h, best_psl

def _logistic_init(r, n, seed):
    """
    Generate a sequence of +/- 1 based on the logistic map.
    x_{k+1} = r * x_k * (1 - x_k)
    sign is determined by whether x_k > 0.5
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 0.9) # Initial condition
    signs = np.zeros(n)
    for i in range(n):
        signs[i] = 1.0 if x > 0.5 else -1.0
        x = r * x * (1.0 - x)
        if x <= 0 or x >= 1:
            # Reset to avoid escape or stagnation at 0/1
            x = rng.uniform(0.1, 0.9)
    return signs

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Random
    for i in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 2. Quasi-periodic (Irrational frequencies)
    for trial in range(5):
        phases = rng.uniform(0, 2*np.pi, size=5)
        amps = rng.uniform(0.5, 1.0, size=5)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for k in range(1, 6):
            val += amps[k-1] * np.cos(2*np.pi*k*t + phases[k-1])
        candidates.append(np.sign(val))
        
    # 3. Simple periodic patterns
    for period in [2, 3, 4, 5, 7, 11, 13]:
        pattern = np.sign(np.cos(2*np.pi*np.arange(N)/period))
        candidates.append(pattern)
        
    # 4. Block structure
    block_size = 10
    for i in range(4):
        h = np.zeros(N)
        for b in range(N // block_size):
            h[b*block_size:(b+1)*block_size] = rng.choice([-1.0, 1.0])
        candidates.append(h)
        
    # 5. Logistic map initializations (Chaotic regime)
    # r in [3.5, 4.0] is typically chaotic
    for r in [3.5, 3.7, 3.9, 3.99, 4.0]:
        for seed in [1, 2, 3]:
            candidates.append(_logistic_init(r, N, seed))
            
    # 6. Thue-Morse sequence
    thue = np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(N)])
    candidates.append(thue)
    
    # 7. More diverse random with different densities
    for density in [0.3, 0.5, 0.7]:
        h = np.zeros(N)
        for i in range(N):
            h[i] = 1.0 if rng.random() < density else -1.0
        candidates.append(h)
    
    # Evaluate all candidates and keep the best few for further search
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine the top 5 candidates with fast PSL optimization (Gordon's method)
    for i, (s, h) in enumerate(scored_candidates[:5]):
        if time.time() > deadline - 25:
            break
        # Allocate time based on rank
        time_limit = deadline - 15
        h_refined, s_refined = _fast_psl_optimize(h, max_passes=300, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
        # After single-flip search, try pairwise flips to escape local minima
        if time.time() < deadline - 15:
            acf_best = _compute_acf(best_h)
            h_pair, s_pair = _pairwise_flip_search(best_h, acf_best, max_pairs=100, 
                                                    time_budget=deadline - 15, start_time=start_time)
            if s_pair < best_score:
                best_score = s_pair
                best_h = h_pair.copy()
    
    # If time remains, do a few random restarts with fast PSL optimization
    while time.time() < deadline - 10:
        # Alternate between random and logistic initializations
        if rng.random() < 0.5:
            h_init = rng.choice([-1.0, 1.0], size=N)
        else:
            r = rng.uniform(3.5, 4.0)
            h_init = _logistic_init(r, N, int(rng.integers(0, 10000)))
            
        # Quick optimization with limited time
        time_limit = min(deadline - 8, time.time() + 4)
        h_refined, s_refined = _fast_psl_optimize(h_init, max_passes=200, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()

    if best_h is None:
        # Fallback
        best_h = np.ones(N)
        
    return best_h
