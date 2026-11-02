"""Rewritten program to maximize the score by finding a large Sidon set."""
import numpy as np
import time

N = 600
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    h = np.asarray(h, dtype=np.float64)
    s = np.sum(h)
    if s < 1e-12:
        return 1e10
    # Use FFT for fast convolution
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    h_padded = np.zeros(n_fft)
    h_padded[:N] = h
    conv = np.real(np.fft.ifft(np.fft.fft(h_padded) ** 2))[:2 * N - 1]
    max_conv = np.max(conv)
    return max_conv / (s * s)

def greedy_sidon_set(candidates):
    """
    Greedily construct a Sidon set from the given list of candidates.
    A Sidon set has distinct pairwise sums.
    """
    S = []
    sums = set()
    for i in candidates:
        valid = True
        new_sums = []
        for x in S:
            s_val = x + i
            if s_val in sums:
                valid = False
                break
            new_sums.append(s_val)
        if valid:
            S.append(i)
            sums.update(new_sums)
    return S

def construct():
    start_time = time.time()
    
    best_set = []
    best_size = 0
    
    # 1. Try structured orders
    structured_orders = [
        list(range(N)),
        list(range(N-1, -1, -1)),
        sorted(range(N), key=lambda x: abs(x - N//2))
    ]
    
    for order in structured_orders:
        if time.time() - start_time > DEADLINE - 5:
            break
        S = greedy_sidon_set(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # 2. Try random permutations
    # We can do many of these as it's fast
    rng = np.random.RandomState(42)
    num_randoms = 500
    
    for _ in range(num_randoms):
        if time.time() - start_time > DEADLINE - 5:
            break
        candidates = rng.permutation(N)
        S = greedy_sidon_set(candidates)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            # If we find a very large set, we might break early, 
            # but let's keep searching to see if we can beat it.
            # Theoretical max is ~25. If we hit 26, it's great.
            
    # Construct h
    h = np.zeros(N)
    if best_size > 0:
        for idx in best_set:
            h[idx] = 1.0 / best_size
    else:
        # Fallback to uniform (should not happen)
        h[:] = 1.0 / N
        
    # The score is 1 / value.
    # Value = max_conv / sum^2.
    # For Sidon set, value = 2 / K^2.
    # Score = K^2 / 2.
    
    return h.tolist()
# EVOLVE-BLOCK-END
