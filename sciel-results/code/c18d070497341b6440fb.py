import numpy as np
import time

N = 400

def _linear_acf(h):
    """Compute linear autocorrelation for lags 0..N-1 via FFT."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n - 1:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return acf[:n]

def _score(h):
    """Compute the max absolute autocorrelation for lags 1 to N-1 (linear)."""
    acf = _linear_acf(h)
    if len(acf) <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:])))

def _psl_value(acf, n):
    """Get PSL value from acf array (max abs at non-zero lags)."""
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))

def _compute_all_deltas(h, n):
    """Compute delta vectors for all positions i in O(N^2) using convolution."""
    # For each position i, delta[i][k] = -2*h[i] * (h[i-k] + h[i+k]) for k=1..n-1
    # We can compute this efficiently using the autocorrelation structure
    
    # Method: For each lag k, compute the contribution to all positions
    # delta[i][k] = -2*h[i] * (h[i-k] + h[i+k])
    
    # Build a matrix where D[i,k] = h[i-k] + h[i+k] for valid indices
    ks = np.arange(1, n)  # lags 1 to n-1
    i_idx = np.arange(n)
    
    # Create index matrices
    i_grid, k_grid = np.meshgrid(i_idx, ks, indexing='ij')
    idx1 = i_grid - k_grid
    idx2 = i_grid + k_grid
    
    # Valid masks
    valid1 = (idx1 >= 0) & (idx1 < n)
    valid2 = (idx2 >= 0) & (idx2 < n)
    
    # Compute neighbor sums
    neighbor_sum = np.zeros((n, n-1))
    h_flat = h
    idx1_clipped = np.clip(idx1, 0, n-1)
    idx2_clipped = np.clip(idx2, 0, n-1)
    
    neighbor_sum = np.where(valid1, h_flat[idx1_clipped], 0.0) + np.where(valid2, h_flat[idx2_clipped], 0.0)
    
    # delta[i,k] = -2 * h[i] * neighbor_sum[i,k]
    deltas = -2.0 * h[:, np.newaxis] * neighbor_sum
    
    return deltas

def _gradient_step(h, acf, deltas=None):
    """
    Find the single best sign flip that reduces the linear PSL.
    """
    n = len(h)
    current_psl = _psl_value(acf, n)
    
    if deltas is None:
        deltas = _compute_all_deltas(h, n)
    
    acf_lags = acf[1:n]
    
    # For each position i, compute new_acf_lags = acf_lags + deltas[i]
    # and find the max abs value
    new_acf_matrix = acf_lags[np.newaxis, :] + deltas  # shape (n, n-1)
    new_psls = np.max(np.abs(new_acf_matrix), axis=1)
    
    best_i = int(np.argmin(new_psls))
    best_new_psl = float(new_psls[best_i])
    
    return best_i, best_new_psl, current_psl

def _fast_psl_optimize(h, max_passes=200, time_budget=None, start_time=None):
    """
    Fast PSL optimization using Gordon's gradient method with vectorized delta computation.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    acf = _linear_acf(h)
    best_psl = _psl_value(acf, n)
    
    passes = 0
    while passes < max_passes:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        deltas = _compute_all_deltas(h, n)
        i, new_psl, cur_psl = _gradient_step(h, acf, deltas=deltas)
        
        if i == -1 or new_psl >= cur_psl:
            break
        
        h[i] = -h[i]
        passes += 1
        
        acf = _linear_acf(h)
        cur_psl = _psl_value(acf, n)
        
        if cur_psl < best_psl:
            best_psl = cur_psl
            best_h = h.copy()
        
        if best_psl < 1e-10:
            break
    
    return best_h, best_psl

def _chirp_candidates(rng, n_chirps=8):
    """
    Generate chirp-based initializations.
    A chirp signal is exp(2j*pi*(a*t^2 + b*t)) for t in [0,1).
    The sign of the real part (or the signal itself) gives a +/-1 sequence.
    We optimize over a and b to get good spectral properties.
    """
    candidates = []
    
    # Zadoff-Chu-like: a=1, b=0, but we vary
    for a in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
            t = np.linspace(0, 1, N, endpoint=False)
            # Quadratic chirp
            phase = 2 * np.pi * (a * t**2 + b * t)
            # Use sign of real part
            h = np.sign(np.cos(phase))
            # Avoid zeros
            h[h == 0] = 1.0
            candidates.append(h)
    
    # Linear chirp (frequency sweep)
    for a in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        t = np.linspace(0, 1, N, endpoint=False)
        phase = 2 * np.pi * a * t
        h = np.sign(np.cos(phase))
        h[h == 0] = 1.0
        candidates.append(h)
    
    # Combined: sum of two chirps with different parameters
    for _ in range(n_chirps):
        a1 = rng.uniform(0.5, 5.0)
        b1 = rng.uniform(0.0, 1.0)
        a2 = rng.uniform(0.5, 5.0)
        b2 = rng.uniform(0.0, 1.0)
        t = np.linspace(0, 1, N, endpoint=False)
        phase1 = 2 * np.pi * (a1 * t**2 + b1 * t)
        phase2 = 2 * np.pi * (a2 * t**2 + b2 * t)
        val = np.cos(phase1) + np.cos(phase2)
        h = np.sign(val)
        h[h == 0] = 1.0
        candidates.append(h)
    
    return candidates

def _simulated_annealing_psl(h, time_budget=30.0, start_time=None, rng=None):
    """
    Simulated annealing on PSL using efficient delta computation.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    acf = _linear_acf(h)
    current_psl = _psl_value(acf, n)
    
    best_h = h.copy()
    best_psl = current_psl
    
    T = 10.0
    T_min = 0.1
    cooling = 0.95
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        i = int(rng.integers(n))
        
        deltas = _compute_all_deltas(h, n)
        delta = deltas[i]
        new_acf_lags = acf[1:n] + delta
        new_psl = float(np.max(np.abs(new_acf_lags)))
        
        delta_psl = new_psl - current_psl
        
        if delta_psl < 0 or np.random.random() < np.exp(-delta_psl / T):
            h[i] = -h[i]
            acf = _linear_acf(h)
            current_psl = _psl_value(acf, n)
            
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
        
        T *= cooling
    
    return best_h, best_psl

def _multi_start_optimize(rng, time_budget, start_time):
    """
    Multi-start optimization: generate diverse initializations,
    refine each with fast PSL, return the best.
    """
    best_h = None
    best_score = float('inf')
    
    # Generate diverse candidates
    candidates = []
    
    # 1. Random
    for i in range(5):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # 2. Chirp-based
    candidates.extend(_chirp_candidates(rng, n_chirps=5))
    
    # 3. Simple periodic patterns
    for period in [2, 3, 4, 5, 7, 11, 13]:
        pattern = np.sign(np.cos(2*np.pi*np.arange(N)/period))
        pattern[pattern == 0] = 1.0
        candidates.append(pattern)
    
    # 4. Block structure
    for block_size in [5, 10, 20]:
        h = np.zeros(N)
        for b in range(N // block_size):
            h[b*block_size:(b+1)*block_size] = rng.choice([-1.0, 1.0])
        candidates.append(h)
    
    # 5. Add some structured patterns based on known good sequences
    # Barker-like patterns
    barker_patterns = [
        [1,1,1,-1,1,-1,1],
        [1,1,1,1,-1,-1,1],
    ]
    for bp in barker_patterns:
        h = np.zeros(N)
        for i in range(N):
            h[i] = bp[i % len(bp)]
        candidates.append(h)
    
    # Evaluate all candidates and keep the best few for further search
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine the top candidates with fast PSL optimization
    top_k = min(10, len(scored_candidates))
    for i, (s, h) in enumerate(scored_candidates[:top_k]):
        if time.time() > time_budget:
            break
        
        # Allocate time based on rank and remaining budget
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        time_limit = min(remaining - 2, start_time + 140)
        
        h_refined, s_refined = _fast_psl_optimize(h, max_passes=300, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # If time remains, do a few random restarts with fast PSL optimization
    while time.time() < time_budget - 5:
        h_init = rng.choice([-1.0, 1.0], size=N)
        time_limit = min(time_budget - 2, time.time() + 3)
        h_refined, s_refined = _fast_psl_optimize(h_init, max_passes=80, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Multi-start optimization
    best_h, best_score = _multi_start_optimize(rng, deadline - 20, start_time)
    
    # Phase 2: If time remains, do SA on the best found so far
    if best_h is not None and time.time() < deadline - 5:
        sa_time = min(15.0, deadline - 5 - (time.time() - start_time))
        if sa_time > 2.0:
            h_sa, s_sa = _simulated_annealing_psl(best_h, time_budget=sa_time, start_time=start_time, rng=rng)
            if s_sa < best_score:
                best_score = s_sa
                best_h = h_sa.copy()
    
    # Phase 3: Final polish with more passes
    if best_h is not None and time.time() < deadline - 2:
        polish_time = min(10.0, deadline - 2 - (time.time() - start_time))
        if polish_time > 1.0:
            h_polished, s_polished = _fast_psl_optimize(best_h, max_passes=500, time_budget=polish_time, start_time=start_time)
            if s_polished < best_score:
                best_score = s_polished
                best_h = h_polished.copy()
    
    if best_h is None:
        # Fallback
        best_h = np.ones(N)
    
    return best_h
