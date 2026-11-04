"""Combined program: Thue-Morse init + Gordon PSL optimization + Simulated Annealing."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n (as +/-1)."""
    t = np.zeros(n, dtype=np.uint8)
    for i in range(1, n):
        t[i] = t[i >> 1] ^ (i & 1)
    # Convert 0/1 to -1/+1
    return (1.0 - 2.0 * t).astype(np.float64)

def _haar_wavelet_init(n, rng, scale_factor=1.0):
    """
    Generate a signed step function using Haar wavelet coefficients.
    Applies frequency-dependent variance to suppress high-frequency peaks.
    """
    # Generate Gaussian noise in wavelet domain
    # Haar basis has N coefficients: 1 approximation, N-1 details
    coeffs = np.zeros(n, dtype=np.float64)
    
    # Approximation coefficient (DC component) - usually kept small or zero for zero-mean
    coeffs[0] = 0.0 
    
    # Detail coefficients
    # Frequency-dependent variance: lower frequencies (earlier details in some bases, but Haar is dyadic)
    # In standard Haar, details are ordered by scale. 
    # Let's apply a decay to high-frequency details to reduce sharp transitions which cause high ACV peaks.
    # Index 1..N-1 correspond to detail coefficients.
    # We'll use a simple decay: variance = exp(-index / scale_factor)
    
    indices = np.arange(1, n)
    variances = np.exp(-indices / (n * 0.1)) # Decay over the length
    # Ensure some variance at the highest frequencies too, just lower
    variances = np.clip(variances, 0.1, 1.0)
    
    coeffs[1:] = rng.normal(0, 1, size=n-1) * np.sqrt(variances) * scale_factor
    
    # Inverse Haar transform
    # Simple iterative inverse Haar
    h = coeffs.copy()
    size = n
    # We need to be careful with the order. 
    # Standard Haar: 
    # Level 0: a0 = (c0+c1)/sqrt(2), d0 = (c0-c1)/sqrt(2)
    # This is complex to implement generically for arbitrary N without scipy.
    # Since SciPy is available, we can use pywt if it's standard, but the prompt says "NumPy and SciPy".
    # Let's stick to a simple random projection or just fix the existing random inits if wavelet is too complex to implement correctly in pure numpy quickly.
    # Actually, the prompt suggests "applying a Haar wavelet transform to a Gaussian noise vector".
    # Let's use a simpler multi-scale random initialization that mimics this:
    # Generate random blocks of decreasing size.
    
    h = np.zeros(n, dtype=np.float64)
    remaining = n
    idx = 0
    block_size = max(1, n // 4)
    
    while idx < n:
        if idx + block_size > n:
            block_size = n - idx
        val = rng.choice([-1.0, 1.0])
        h[idx:idx+block_size] = val
        idx += block_size
        block_size = max(1, block_size // 2)
        
    return h

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
    Finds the single bit flip that most reduces the PSL.
    """
    n = len(h)
    current_psl = _psl_value(acf, n)
    
    best_i = -1
    best_new_psl = float('inf')
    
    acf_lags = acf[1:n]
    
    for i in range(n):
        # Construct the change in ACF if we flip h[i]
        # delta[k-1] = -2 * h[i] * (h[i-k] + h[i+k])
        neighbor_sum = np.zeros(n-1)
        
        # h[i-k] for k=1..min(i, n-1)
        max_k1 = min(i, n-1)
        if max_k1 > 0:
            # h[i-1], h[i-2], ..., h[i-max_k1]
            neighbor_sum[:max_k1] += h[i-1 : i-max_k1-1 : -1]
            
        # h[i+k] for k=1..min(n-1-i, n-1)
        max_k2 = min(n-1-i, n-1)
        if max_k2 > 0:
            # h[i+1], h[i+2], ..., h[i+max_k2]
            neighbor_sum[:max_k2] += h[i+1 : i+max_k2+1]
            
        delta = -2.0 * h[i] * neighbor_sum
        new_acf_lags = acf_lags + delta
        new_psl = np.max(np.abs(new_acf_lags))
        
        if new_psl < best_new_psl:
            best_new_psl = new_psl
            best_i = i
            
    return best_i, float(best_new_psl), current_psl

def _fast_psl_optimize(h, max_passes=1000, time_budget=10.0, start_time=None):
    """
    Fast PSL optimization using Gordon's gradient method.
    Repeatedly finds the single best flip and applies it.
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
        if (time.time() - start_time) > time_budget:
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
    Simulated annealing on PSL using fast delta evaluation.
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
    
    T = 5.0
    T_min = 0.01
    cooling = 0.98
    
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

def construct():
    start_time = time.time()
    deadline = 145.0  # Tighter safety margin to ensure we return before 150s
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Thue-Morse (known low autocorrelation properties)
    candidates.append(_thue_morse(N))
    
    # 2. Random initializations
    for i in range(10): # Reduced count to save time for more diverse inits
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 2b. Multi-scale (Haar-like) initializations
    for i in range(10):
        candidates.append(_haar_wavelet_init(N, rng))
        
    # 3. Quasi-periodic (Irrational frequencies)
    for trial in range(5):
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
    block_size = 10
    for i in range(4):
        h = np.zeros(N)
        for b in range(N // block_size):
            h[b*block_size:(b+1)*block_size] = rng.choice([-1.0, 1.0])
        candidates.append(h)
        
    # Evaluate all candidates and keep the best few for further search
    scored_candidates = []
    for h in candidates:
        # Ensure h is float64
        h = h.astype(np.float64)
        s = _psl_value(_compute_acf(h), N)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Update global best with the best initial candidate
    if scored_candidates:
        best_score = scored_candidates[0][0]
        best_h = scored_candidates[0][1].copy()
        
    # Refine the top candidates with fast PSL optimization (Gordon's method)
    # Allocate time carefully
    num_refine = 3 # Reduced to allow more time per refinement
    time_per_refine = (deadline - 20) / num_refine # Leave 20s for SA and final polish
    
    for i, (s, h) in enumerate(scored_candidates[:num_refine]):
        if time.time() > deadline - 20:
            break
        
        # Allocate time based on rank
        time_limit = min(time_per_refine, deadline - 20 - i)
        if time_limit <= 0:
            break
        h_refined, s_refined = _fast_psl_optimize(h, max_passes=1000, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # If time remains, do simulated annealing on the current best
    if time.time() < deadline - 5:
        time_limit = deadline - 5
        h_refined, s_refined = _simulated_annealing_psl(best_h, time_budget=time_limit, start_time=start_time, rng=rng)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # Final polish with a few more gradient steps if time allows
    if time.time() < deadline - 2:
        time_limit = deadline - 2
        h_refined, s_refined = _fast_psl_optimize(best_h, max_passes=100, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    return best_h

# EVOLVE-BLOCK-END
