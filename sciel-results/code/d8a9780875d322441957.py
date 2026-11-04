"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(h):
    """Compute the max absolute autocorrelation for lags 1 to N-1."""
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return np.max(np.abs(acf[1:N]))


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +1/-1."""
    tm = np.zeros(n, dtype=int)
    for i in range(1, n):
        tm[i] = tm[i >> 1] ^ (i & 1)
    return 1 - 2 * tm


def _local_search(h, time_limit):
    """Fast local search using greedy flipping of individual elements."""
    h = h.copy()
    n = len(h)
    current_score = _score(h)
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        best_delta = 0
        best_pos = -1
        best_new_score = current_score
        
        for j in range(n):
            h[j] = -h[j]
            new_score = _score(h)
            if new_score < best_new_score - 1e-9:
                best_new_score = new_score
                best_pos = j
                best_delta = new_score - current_score
            h[j] = -h[j]  # revert
        
        if best_pos != -1:
            h[best_pos] = -h[best_pos]
            current_score = best_new_score
            improved = True
            
    return h, current_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Strategy:
    # 1. Start with Thue-Morse sequence (known to have flat autocorrelation)
    # 2. Generate diverse initial candidates (random, quasi-periodic, periodic, block)
    # 3. Use fast local search to refine the best candidates
    
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Thue-Morse sequence (structurally different, known good autocorrelation)
    candidates.append(_thue_morse(N).astype(float))
    
    # 2. Random candidates
    for i in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 3. Quasi-periodic (Irrational frequencies)
    for trial in range(5):
        phases = rng.uniform(0, 2 * np.pi, size=5)
        amps = rng.uniform(0.5, 1.0, size=5)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for k in range(1, 6):
            val += amps[k - 1] * np.cos(2 * np.pi * k * t + phases[k - 1])
        candidates.append(np.sign(val))
        
    # 4. Simple periodic patterns
    for period in [2, 3, 4, 5, 7, 11, 13]:
        pattern = np.sign(np.cos(2 * np.pi * np.arange(N) / period))
        candidates.append(pattern)
        
    # 5. Block structure
    block_size = 10
    for i in range(4):
        h = np.zeros(N)
        for b in range(N // block_size):
            h[b * block_size:(b + 1) * block_size] = rng.choice([-1.0, 1.0])
        candidates.append(h)

    # Evaluate all candidates and keep the best few for further search
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine the top 5 candidates with local search
    for i, (s, h) in enumerate(scored_candidates[:5]):
        if time.time() > deadline - 20:
            break
        time_limit = deadline - 10
        h_refined, s_refined = _local_search(h, time_limit)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # If time remains, do a few random restarts with local search
    while time.time() < deadline - 10:
        h_init = rng.choice([-1.0, 1.0], size=N)
        time_limit = min(deadline - 5, time.time() + 5)
        h_refined, s_refined = _local_search(h_init, time_limit)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    if best_h is None:
        # Fallback
        best_h = np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
