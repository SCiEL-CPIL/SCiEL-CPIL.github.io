"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(h):
    """
    Compute the upper bound on C3.
    The task asks for the upper bound on C3 from a signed step function.
    Based on the inspirations, this is calculated as dx * max |autoconvolution|.
    dx = 0.5 / N.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    dx = 0.5 / n
    
    # Linear autoconvolution via FFT
    # Pad to length 2*n-1
    fft_size = 1
    while fft_size < 2 * n - 1:
        fft_size *= 2
        
    H = np.fft.rfft(h, n=fft_size)
    # Power spectrum
    P = np.abs(H) ** 2
    # Inverse FFT to get autocorrelation (circular, but padded enough for linear)
    acf = np.fft.irfft(P, n=fft_size)
    
    # The linear autocorrelation is in acf[0:n] (positive lags) and acf[fft_size-n+1:fft_size] (negative lags)
    # We want max absolute value for lags 1 to n-1.
    # acf[0] is lag 0.
    # acf[1:n] are lags 1 to n-1.
    # acf[fft_size-n+1 : fft_size] are lags -(n-1) to -1.
    
    # Due to symmetry, max abs of positive lags is enough, but let's be safe and check both if needed.
    # Usually acf is real and symmetric.
    # Let's just take the max of the non-zero lags.
    # The peak at lag 0 is n * (mean^2) roughly, which is large. We exclude it.
    
    # Extract non-zero lags
    # Positive lags: 1 to n-1
    pos_lags = acf[1:n]
    # Negative lags: corresponding indices
    # For circular correlation of length L, lag k corresponds to index k.
    # Lag -k corresponds to index L-k.
    # Here L = fft_size.
    # We only care about lags up to n-1.
    # So indices: 1..n-1 and fft_size-(n-1)..fft_size-1.
    neg_lags = acf[fft_size-(n-1):fft_size]
    
    max_abs = np.max(np.abs(np.concatenate([pos_lags, neg_lags])))
    
    return dx * max_abs

def _local_search(h, deadline, best_score, best_h):
    """
    Greedy local search.
    Tries flipping each bit. If it improves the score, keep it.
    Repeats until no improvement or time limit.
    """
    h = np.array(h, dtype=np.float64)
    current_score = _score(h)
    
    if current_score < best_score:
        best_score = current_score
        best_h = h.copy()
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try flipping each bit
        for i in range(N):
            if time.time() > deadline:
                break
            old_val = h[i]
            h[i] = -old_val
            new_score = _score(h)
            
            if new_score < current_score - 1e-12:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
                improved = True
            else:
                h[i] = old_val
                
    return best_score, best_h

def construct():
    start_time = time.time()
    # Safety margin: 150s limit. Use 140s.
    deadline = start_time + 140.0
    
    best_score = float('inf')
    best_h = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Thue-Morse
    tm = []
    cur = 0
    for i in range(N):
        # Thue-Morse: number of 1s in binary representation of i, mod 2
        # Or recursive generation
        pass
    # Simpler Thue-Morse generation
    seq = [1]
    while len(seq) < N:
        seq += [-x for x in seq]
    candidates.append(np.array(seq[:N], dtype=np.float64))
    
    # 2. Random
    for _ in range(5):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 3. Quasi-periodic
    for trial in range(5):
        phases = rng.uniform(0, 2*np.pi, size=5)
        amps = rng.uniform(0.5, 1.0, size=5)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for k in range(1, 6):
            val += amps[k-1] * np.cos(2*np.pi*k*t + phases[k-1])
        candidates.append(np.sign(val))
        
    # 4. Periodic patterns
    for period in [2, 3, 4, 5, 7, 11, 13]:
        pattern = np.sign(np.cos(2*np.pi*np.arange(N)/period))
        candidates.append(pattern)
        
    # 5. Block structures
    block_size = 10
    for _ in range(4):
        h = np.zeros(N)
        for b in range(N // block_size):
            h[b*block_size:(b+1)*block_size] = rng.choice([-1.0, 1.0])
        candidates.append(h)
        
    # Evaluate and sort candidates
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
        
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine top candidates
    # Allocate time to top 5
    # If we have 140s, spend most on the best few.
    for i, (s, h) in enumerate(scored_candidates[:5]):
        if time.time() > deadline - 5:
            break
        # Give more time to better candidates? Or equal time?
        # Let's just run local search until deadline - 5 for each? No, that's too much.
        # Let's run local search with a sub-deadline.
        # If we have 5 candidates, we can split time.
        # But local search converges quickly often.
        # Let's just run until deadline - 5 for the first one, then less for others?
        # Actually, the first one is likely the best.
        
        # Let's run local search for each, but stop if we run out of time.
        # We pass the global deadline. The function checks time.
        best_score, best_h = _local_search(h, deadline - 2.0, best_score, best_h)
        
    # If time remains, try random restarts
    while time.time() < deadline - 5:
        h_init = rng.choice([-1.0, 1.0], size=N)
        # Quick local search
        # Limit time for this restart to avoid getting stuck?
        # The _local_search function checks deadline.
        # But if we have many bad restarts, we waste time.
        # Let's limit each restart to, say, 2 seconds?
        # Or just let it run until deadline.
        # If it improves, great. If not, it will stop when no improvement.
        # But if it finds a local min, it stops.
        
        # To be safe, let's create a sub-deadline for each random restart
        # so we don't spend 100s on a bad one.
        sub_deadline = min(deadline - 2.0, time.time() + 5.0)
        best_score, best_h = _local_search(h_init, sub_deadline, best_score, best_h)
        
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
