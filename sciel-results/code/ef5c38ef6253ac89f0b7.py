import numpy as np
import time

N = 400

def _compute_score(h):
    """
    Computes the raw score: max(|autoconv(h)|[1:]) / (sum(h))^2
    Assumes lag 0 is excluded based on deduction that seed is not global min otherwise.
    """
    S = np.sum(h)
    if S == 0:
        return float('inf')
    
    # Linear autoconvolution
    # Use FFT for speed
    n_fft = 2 * N
    F = np.fft.fft(h, n=n_fft)
    A = np.fft.ifft(F * F).real[:N]
    
    # Exclude lag 0
    max_abs_conv = np.max(np.abs(A[1:]))
    
    return max_abs_conv / (S * S)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Strategy:
    # 1. Initialize with a Legendre symbol sequence (length p ~ N) to get low initial correlation.
    # 2. Pad/truncate to N.
    # 3. Use local search (sign flips) to optimize the ratio.
    
    # Find prime p close to N
    p = 397 # Prime less than 400
    # Legendre symbol for 1..p-1, 0 for 0
    # chi(x) = 1 if x is quadratic residue mod p, -1 if not, 0 if x=0
    def legendre(a, p):
        if a % p == 0:
            return 0
        return pow(a, (p-1)//2, p) == 1 and 1 or -1
    
    # Generate sequence
    h = np.zeros(N, dtype=float)
    for i in range(p):
        h[i] = 1.0 if legendre(i, p) == 1 else -1.0
    # Pad remaining with 1s to increase sum initially
    h[p:] = 1.0
    
    # Initial score
    current_score = _compute_score(h)
    
    # Local search
    # We will try flipping individual bits and pairs of bits
    # To speed up, we can just do single flips first, then maybe pairs
    
    best_h = h.copy()
    best_score = current_score
    
    # Simple hill climbing with random restarts if stuck?
    # Given time limit, a simple iterative improvement is safe.
    
    step = 1
    while time.time() < deadline:
        # Propose a random flip
        idx = np.random.randint(0, N)
        new_h = best_h.copy()
        new_h[idx] *= -1
        
        # Evaluate
        new_score = _compute_score(new_h)
        
        if new_score < best_score:
            best_score = new_score
            best_h = new_h
        else:
            # If no improvement, try a pair flip to escape local minima
            if np.random.rand() < 0.1:
                idx2 = np.random.randint(0, N)
                if idx2 != idx:
                    new_h2 = best_h.copy()
                    new_h2[idx] *= -1
                    new_h2[idx2] *= -1
                    new_score2 = _compute_score(new_h2)
                    if new_score2 < best_score:
                        best_score = new_score2
                        best_h = new_h2
    
    return best_h

# EVOLVE-BLOCK-END
