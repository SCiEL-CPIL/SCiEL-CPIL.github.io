import numpy as np
import time

N = 400

def _compute_l3_bound(h):
    """
    Compute the L3 norm upper bound on C3.
    The autoconvolution is computed via FFT, and we take the sum of cubes
    of absolute values at non-zero lags, scaled by dx.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    dx = 0.5 / n
    
    # Compute autoconvolution via FFT
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    
    # The continuous autoconvolution values are dx * acf[k]
    # L3 norm: sum of |dx * acf[k]|^3 for k=1 to n-1
    # = dx^3 * sum(|acf[k]|^3)
    acf_lags = acf[1:n]
    l3_val = dx ** 3 * np.sum(np.abs(acf_lags) ** 3)
    
    return float(l3_val)

def _compute_acf_full(h):
    """Compute full autocorrelation via FFT, return acf array."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return acf

def _l3_from_acf(acf, n):
    """Compute L3 bound from precomputed acf."""
    dx = 0.5 / n
    acf_lags = acf[1:n]
    return float(dx ** 3 * np.sum(np.abs(acf_lags) ** 3))

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1 values."""
    seq = np.array([1.0])
    while len(seq) < n:
        seq = np.concatenate([seq, -seq])
    return seq[:n].astype(np.float64)

def _sa_l3_optimize(h, time_budget=10.0, start_time=None, T0=1.0, T_end=1e-8, cooling_rate=0.9995):
    """
    Simulated Annealing optimization for L3 norm objective.
    Uses incremental ACF updates for efficiency.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    acf = _compute_acf_full(h)
    best_l3 = _l3_from_acf(acf, n)
    current_l3 = best_l3
    
    T = T0
    rng = np.random.default_rng(12345)
    
    # Precompute: for flipping h[i], the change in acf[k] is:
    # delta_acf[k] = -2*h[i]*(h[i-k] + h[i+k]) for valid indices
    # We'll compute this on the fly for the selected index
    
    while (time.time() - start_time) < time_budget and T > T_end:
        # Pick a random index to flip
        i = rng.integers(0, n)
        
        old_val = h[i]
        
        # Compute delta in acf for flipping h[i]
        # acf[k] = sum_j h[j]*h[j+k] (for k>=1)
        # Flipping h[i]: h[i] -> -h[i]
        # Change in acf[k]: 
        #   if i-k >= 0: term h[i-k]*h[i] changes from h[i-k]*old to h[i-k]*(-old) -> delta = -2*old*h[i-k]
        #   if i+k < n: term h[i]*h[i+k] changes from old*h[i+k] to (-old)*h[i+k] -> delta = -2*old*h[i+k]
        # So delta_acf[k] = -2*old*(h[i-k] + h[i+k]) where valid terms are included
        
        delta_acf = np.zeros(n - 1)
        # For lag k (1-indexed in acf[1:n], 0-indexed in delta_acf):
        # k corresponds to acf[k+1]
        # h[i-k] valid if i-k >= 0, i.e., k <= i
        # h[i+k] valid if i+k < n, i.e., k < n-i
        for k in range(1, n):
            contrib = 0.0
            if i - k >= 0:
                contrib += h[i - k]
            if i + k < n:
                contrib += h[i + k]
            delta_acf[k - 1] = -2.0 * old_val * contrib
        
        # New acf lags
        new_acf_lags = acf[1:n] + delta_acf
        new_l3 = float(dx ** 3 * np.sum(np.abs(new_acf_lags) ** 3)) if False else None
        
        # Actually, let's compute properly
        dx = 0.5 / n
        new_l3 = float(dx ** 3 * np.sum(np.abs(new_acf_lags) ** 3))
        
        # Acceptance criterion
        delta = new_l3 - current_l3
        if delta < 0 or (T > T_end and rng.random() < np.exp(-delta / T)):
            # Accept the move
            h[i] = -old_val
            
            # Update ACF
            acf[1:n] += delta_acf
            
            current_l3 = new_l3
            if current_l3 < best_l3:
                best_l3 = current_l3
                best_h = h.copy()
        
        T *= cooling_rate
        
        if best_l3 < 1e-15:
            break
            
    return best_h, best_l3

def _logistic_init(r, n, seed, init_x=None):
    """
    Generate a sequence of +/- 1 based on the logistic map.
    """
    rng = np.random.default_rng(seed)
    if init_x is None:
        x = rng.uniform(0.1, 0.9)
    else:
        x = init_x
    signs = np.zeros(n)
    for i in range(n):
        signs[i] = 1.0 if x > 0.5 else -1.0
        x = r * x * (1.0 - x)
        if x <= 0 or x >= 1:
            x = rng.uniform(0.1, 0.9)
    return signs

def _construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Thue-Morse (structured, low autocorrelation)
    candidates.append(_thue_morse(N))
    
    # 2. Random sequences
    for i in range(15):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 3. Quasi-periodic (Irrational frequencies)
    for trial in range(8):
        phases = rng.uniform(0, 2*np.pi, size=5)
        amps = rng.uniform(0.5, 1.0, size=5)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for k in range(1, 6):
            val += amps[k-1] * np.cos(2*np.pi*k*t + phases[k-1])
        candidates.append(np.sign(val))
        
    # 4. Simple periodic patterns
    for period in [2, 3, 4, 5, 7, 11, 13]:
        pattern = np.sign(np.cos(2*np.pi*np.arange(N)/period))
        candidates.append(pattern)
        
    # 5. Block structure
    for block_size in [5, 10, 20]:
        for i in range(5):
            h = np.zeros(N)
            for b in range(N // block_size):
                h[b*block_size:(b+1)*block_size] = rng.choice([-1.0, 1.0])
            candidates.append(h)
            
    # 6. Logistic map initializations
    for seed in [42, 100, 200]:
        for trial in range(5):
            r = rng.uniform(3.7, 4.0)
            init_x = rng.uniform(0.1, 0.9)
            h = _logistic_init(r, N, seed + trial, init_x)
            candidates.append(h)
    
    # Evaluate all candidates
    scored_candidates = []
    for h in candidates:
        s = _compute_l3_bound(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine the top candidates with SA optimization
    top_k = min(8, len(scored_candidates))
    for i in range(top_k):
        if time.time() > deadline - 10:
            break
        
        s, h = scored_candidates[i]
        if s >= best_score:
            continue
            
        time_left = deadline - time.time()
        candidates_left = top_k - i
        time_per_candidate = max(1.0, min(20.0, time_left / max(1, candidates_left)))
        
        h_refined, s_refined = _sa_l3_optimize(h, time_budget=time_per_candidate, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # If time remains, do random restarts with SA
    while time.time() < deadline - 5:
        h_init = rng.choice([-1.0, 1.0], size=N)
        s_init = _compute_l3_bound(h_init)
        
        if s_init >= best_score * 1.5:  # Only refine if reasonably close
            continue
            
        time_left = deadline - time.time()
        time_per_candidate = max(1.0, min(15.0, time_left / 8.0))
        
        h_refined, s_refined = _sa_l3_optimize(h_init, time_budget=time_per_candidate, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()

    if best_h is None:
        best_h = np.ones(N)
        
    return best_h

def construct():
    return _construct()
