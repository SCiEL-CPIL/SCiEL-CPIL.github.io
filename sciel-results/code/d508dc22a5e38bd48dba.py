"""
Program to minimize the peak autoconvolution (C3 upper bound) of a signed step function.
Combines diverse initializations (random, Legendre, m-sequence) with
Gordon's fast PSL minimization and simulated annealing refinement.
"""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _compute_acf(h):
    """
    Compute the full autocorrelation via FFT.
    For a sequence h of length n, the autocorrelation at lag k is
    sum_{i} h[i] * h[i+k] (circular, zero-padded).
    """
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return acf

def _psl_value(acf, n):
    """
    Get the Peak Side Lobe (PSL) value from the autocorrelation array.
    This is the maximum absolute value at non-zero lags (1 to n-1).
    """
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))

def _score(h):
    """
    Compute the score: max absolute autocorrelation for lags 1 to N-1.
    """
    acf = _compute_acf(h)
    return _psl_value(acf, len(h))

def _gradient_step(h, acf):
    """
    Gordon's fast PSL minimization step.
    Finds the single flip that most reduces the PSL.
    Returns (best_index, new_psl, current_psl).
    """
    n = len(h)
    current_psl = _psl_value(acf, n)
    
    best_i = -1
    best_new_psl = float('inf')
    
    acf_lags = acf[1:n]
    
    for i in range(n):
        # Compute the change in ACF if we flip h[i]
        # delta[k] = -2 * h[i] * (h[i-k] + h[i+k]) for valid indices
        neighbor_sum = np.zeros(n-1)
        
        # h[i-k] for k=1..min(i, n-1)
        max_k1 = min(i, n-1)
        if max_k1 > 0:
            # indices: i-1, i-2, ..., i-max_k1
            neighbor_sum[:max_k1] += h[i-1 : i-max_k1-1 : -1]
            
        # h[i+k] for k=1..min(n-1-i, n-1)
        max_k2 = min(n-1-i, n-1)
        if max_k2 > 0:
            # indices: i+1, i+2, ..., i+max_k2
            neighbor_sum[:max_k2] += h[i+1 : i+max_k2+1]
            
        delta = -2.0 * h[i] * neighbor_sum
        new_acf_lags = acf_lags + delta
        new_psl = np.max(np.abs(new_acf_lags))
        
        if new_psl < best_new_psl:
            best_new_psl = new_psl
            best_i = i
            
    return best_i, float(best_new_psl), current_psl

def _fast_psl_optimize(h, max_passes=500, time_budget=None, start_time=None):
    """
    Fast PSL optimization using Gordon's gradient method.
    Repeatedly finds the best flip and applies it.
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
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        i, new_psl, cur_psl = _gradient_step(h, acf)
        
        if i == -1 or new_psl >= cur_psl:
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
    
    acf = _compute_acf(h)
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
        
        # Compute delta for this specific i
        neighbor_sum = np.zeros(n-1)
        max_k1 = min(i, n-1)
        if max_k1 > 0:
            neighbor_sum[:max_k1] += h[i-1 : i-max_k1-1 : -1]
        max_k2 = min(n-1-i, n-1)
        if max_k2 > 0:
            neighbor_sum[:max_k2] += h[i+1 : i+max_k2+1]
            
        delta = -2.0 * h[i] * neighbor_sum
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

def _legendre_sequence(n):
    """
    Generate a Legendre sequence of length n.
    """
    p = 2 * n + 1
    while True:
        is_prime = True
        for i in range(2, int(np.sqrt(p)) + 1):
            if p % i == 0:
                is_prime = False
                break
        if is_prime:
            break
        p += 2 if p > 2 else 1
    
    seq = np.zeros(n)
    for i in range(n):
        val = pow(i + 1, (p - 1) // 2, p)
        if val == 1:
            seq[i] = 1.0
        elif val == p - 1:
            seq[i] = -1.0
        else:
            seq[i] = 1.0
    
    return seq

def _m_sequence(n):
    """
    Generate an m-sequence of length n.
    """
    degree = 9
    poly = (1 << degree) | (1 << 4) | 1
    
    state = 1
    seq = []
    for _ in range(2**degree - 1):
        seq.append(1.0 if (state & 1) == 1 else -1.0)
        lsb = state & 1
        state >>= 1
        if lsb:
            state ^= poly
    
    return np.array(seq[:n])

def construct():
    """
    Construct a signed step function with N intervals to minimize the
    autoconvolution upper bound.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Random initializations
    for i in range(8):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 2. Legendre sequence
    candidates.append(_legendre_sequence(N))
    
    # 3. M-sequence
    candidates.append(_m_sequence(N))
    
    # 4. Quasi-periodic patterns
    for trial in range(4):
        phases = rng.uniform(0, 2*np.pi, size=5)
        amps = rng.uniform(0.5, 1.0, size=5)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for k in range(1, 6):
            val += amps[k-1] * np.cos(2*np.pi*k*t + phases[k-1])
        candidates.append(np.sign(val))
        
    # 5. Simple periodic patterns
    for period in [3, 5, 7, 11]:
        pattern = np.sign(np.cos(2*np.pi*np.arange(N)/period))
        candidates.append(pattern)
        
    # Evaluate all candidates
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine top candidates with fast PSL optimization
    top_k = 6
    for i, (s, h) in enumerate(scored_candidates[:top_k]):
        if time.time() > deadline - 15:
            break
        
        time_limit = deadline - 10
        h_refined, s_refined = _fast_psl_optimize(
            h, max_passes=300, time_budget=time_limit, start_time=start_time
        )
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # If time remains, do simulated annealing on the best candidate
    if best_h is not None and time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 5
        h_sa, s_sa = _simulated_annealing_psl(
            best_h, time_budget=remaining_time, start_time=start_time, rng=rng
        )
        if s_sa < best_score:
            best_score = s_sa
            best_h = h_sa.copy()
    
    # Fallback if nothing was found
    if best_h is None:
        best_h = scored_candidates[0][1]
    
    return best_h

# EVOLVE-BLOCK-END
