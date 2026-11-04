import numpy as np
import time

N = 400

def _autocorr_max_abs(x):
    """Compute max |autocorrelation| for lags 1..N-1 of a ±1 sequence."""
    # Use FFT for efficient autocorrelation
    n = len(x)
    # Pad to next power of 2 for FFT efficiency
    size = 1
    while size < 2 * n:
        size *= 2
    X = np.fft.fft(x, size)
    acf = np.fft.ifft(X * np.conj(X)).real[:n]
    # acf[0] is the DC term (sum of squares = n), we want lags 1..n-1
    return np.max(np.abs(acf[1:]))

def _score(x):
    """The score is the upper bound on C3, which is the peak autoconvolution."""
    return _autocorr_max_abs(x)

def construct():
    deadline = time.time() + 100.0  # Safety margin before 120s limit
    
    # Generate initial candidates
    candidates = []
    
    # 1. Random ±1 sequences
    rng = np.random.default_rng(42)
    for _ in range(10):
        candidates.append(rng.choice([-1, 1], size=N))
    
    # 2. Legendre sequence of length 397 (prime), then extend to 400
    p = 397
    # Legendre symbol: for prime p, x^((p-1)/2) mod p is 1 if x is quadratic residue, -1 otherwise
    # Legendre sequence: a[0] = 1, a[i] = legendre_symbol(i, p) for i=1..p-1
    def legendre_symbol(a, p):
        if a % p == 0:
            return 0
        return pow(a, (p-1)//2, p)
    
    leg = np.zeros(p, dtype=np.int8)
    leg[0] = 1
    for i in range(1, p):
        ls = legendre_symbol(i, p)
        leg[i] = 1 if ls == 1 else -1
    
    # Extend to 400: repeat first 3 elements
    ext = np.zeros(N, dtype=np.int8)
    ext[:p] = leg
    ext[p:] = leg[:N-p]
    candidates.append(ext)
    
    # 3. Greedy construction: build sequence to minimize autocorrelation
    def greedy_construct():
        x = np.zeros(N, dtype=np.int8)
        x[0] = 1
        for i in range(1, N):
            # Try both +1 and -1, pick the one that minimizes max autocorrelation
            best_val = None
            best_score = np.inf
            for v in [1, -1]:
                x[i] = v
                # Only need to check autocorrelations that involve position i
                # But for simplicity, compute full score (slow but correct)
                s = _autocorr_max_abs(x)
                if s < best_score:
                    best_score = s
                    best_val = v
            x[i] = best_val
        return x
    
    # Greedy is too slow for N=400, skip it. Use a simpler heuristic instead.
    # 4. Alternating pattern
    candidates.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.int8))
    
    # 5. Block pattern: +++++-----+++++-----...
    block = 4
    pattern = np.array([1]*block + [-1]*block)
    reps = N // len(pattern) + 1
    candidates.append(np.tile(pattern, reps)[:N].astype(np.int8))
    
    # Evaluate all candidates
    best_x = None
    best_score = np.inf
    for c in candidates:
        s = _score(c)
        if s < best_score:
            best_score = s
            best_x = c.copy()
    
    # Local search: flip individual elements
    current_x = best_x.copy()
    current_score = best_score
    
    # Simulated annealing-like local search
    temperature = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        improved = False
        # Randomly select positions to try flipping
        for _ in range(50):  # Try 50 random flips per iteration
            if time.time() >= deadline:
                break
            idx = rng.integers(0, N)
            old_val = current_x[idx]
            current_x[idx] = -old_val
            new_score = _score(current_x)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_x = current_x.copy()
            else:
                # Accept with some probability (annealing)
                delta = new_score - current_score
                if delta > 0 and rng.random() < np.exp(-delta / max(temperature, 0.001)):
                    current_score = new_score
                else:
                    current_x[idx] = old_val
        
        temperature *= cooling
    
    return best_x
