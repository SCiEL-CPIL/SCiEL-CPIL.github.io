"""Combined program: Thue-Morse initialization + fast local search with ACF delta updates."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    h = np.zeros(n)
    for i in range(n):
        # Count trailing 1s in binary representation
        c = 0
        x = i
        while x & 1:
            c += 1
            x >>= 1
        h[i] = 1.0 if c % 2 == 0 else -1.0
    return h

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

def _local_search_fast(h, time_limit):
    """
    Fast local search using ACF delta updates.
    For flipping h[i], the change in ACF at lag k is:
      delta[k] = -2 * h[i] * (h[i-k] + h[i+k])  (where indices are valid)
    We track the current ACF and update it incrementally.
    """
    n = len(h)
    h = h.copy()
    acf = _compute_acf(h)
    current_psl = float(np.max(np.abs(acf[1:n])))
    
    # Precompute FFT size for potential re-verification
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        best_pos = -1
        best_new_psl = current_psl
        
        # For each position i, compute the new PSL if we flip h[i]
        # The change in ACF for flipping position i:
        #   For lag k (1 <= k <= n-1):
        #     The terms involving h[i] are:
        #       h[i] * h[i+k] (if i+k < n)  -> becomes -h[i]*h[i+k] -> change: -2*h[i]*h[i+k]
        #       h[i-k] * h[i] (if i-k >= 0) -> becomes h[i-k]*(-h[i]) -> change: -2*h[i]*h[i-k]
        #     So delta_acf[k] = -2 * h[i] * (h[i-k] if i>=k else 0) + (-2 * h[i] * h[i+k] if i+k<n else 0)
        #     = -2 * h[i] * (h[i-k] + h[i+k]) with appropriate bounds
        
        for i in range(n):
            # Compute delta for all lags
            # delta[k] for k=1..n-1
            # h[i-k] valid when i >= k, i.e., k <= i
            # h[i+k] valid when i+k < n, i.e., k < n-i
            
            # Vectorized computation of delta
            # For lag k:
            #   left = h[i-k] if k <= i else 0
            #   right = h[i+k] if i+k < n else 0
            #   delta[k] = -2 * h[i] * (left + right)
            
            # Build delta array efficiently
            # left part: lags 1..i contribute -2*h[i]*h[i-k]
            # right part: lags 1..(n-1-i) contribute -2*h[i]*h[i+k]
            
            delta = np.zeros(n)  # delta[1..n-1]
            
            if i > 0:
                # lags 1 to i: left = h[i-k]
                # k ranges 1..i, h[i-k] = h[i-1], h[i-2], ..., h[0]
                # So delta[k] += -2*h[i]*h[i-k] for k=1..i
                # This is -2*h[i] * h[0:i] reversed? No.
                # For k=1: h[i-1], k=2: h[i-2], ..., k=i: h[0]
                # So delta[1:i+1] += -2*h[i] * h[i-1::-1]
                delta[1:i+1] += -2.0 * h[i] * h[i-1::-1]
            
            if i + 1 < n:
                # lags 1 to (n-1-i): right = h[i+k]
                # For k=1: h[i+1], k=2: h[i+2], ..., k=n-1-i: h[n-1]
                # So delta[1:n-i] += -2*h[i] * h[i+1:n]
                delta[1:n-i] += -2.0 * h[i] * h[i+1:n]
            
            # New ACF
            new_acf = acf + delta
            new_psl = float(np.max(np.abs(new_acf[1:n])))
            
            if new_psl < best_new_psl - 1e-9:
                best_new_psl = new_psl
                best_pos = i
                best_delta = delta
                best_new_acf = new_acf
        
        if best_pos != -1:
            # Apply the flip
            h[best_pos] = -h[best_pos]
            acf = best_new_acf.copy()
            current_psl = best_new_psl
            improved = True
    
    return h, current_psl

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Thue-Morse (known for flat autocorrelation)
    candidates.append(_thue_morse(N))
    
    # 2. Random
    for i in range(8):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # 3. Quasi-periodic
    for trial in range(4):
        phases = rng.uniform(0, 2 * np.pi, size=5)
        amps = rng.uniform(0.5, 1.0, size=5)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for k in range(1, 6):
            val += amps[k-1] * np.cos(2 * np.pi * k * t + phases[k-1])
        candidates.append(np.sign(val))
    
    # 4. Simple periodic patterns
    for period in [3, 5, 7, 11, 13]:
        pattern = np.sign(np.cos(2 * np.pi * np.arange(N) / period))
        candidates.append(pattern)
    
    # 5. Block structure
    for block_size in [8, 16, 25]:
        h = np.zeros(N)
        n_blocks = N // block_size
        for b in range(n_blocks):
            h[b*block_size:(b+1)*block_size] = rng.choice([-1.0, 1.0])
        candidates.append(h)
    
    # Evaluate all candidates
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine top candidates with fast local search
    for i, (s, h) in enumerate(scored_candidates[:5]):
        if time.time() > deadline - 15:
            break
        time_limit = deadline - 10
        h_refined, s_refined = _local_search_fast(h, time_limit)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # Random restarts if time remains
    while time.time() < deadline - 15:
        h_init = rng.choice([-1.0, 1.0], size=N)
        time_limit = min(deadline - 5, time.time() + 8)
        h_refined, s_refined = _local_search_fast(h_init, time_limit)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
