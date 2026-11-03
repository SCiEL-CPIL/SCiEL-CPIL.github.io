"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(x):
    """Compute the C3' metric: max abs autoconvolution / (sum of x)^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    # Full linear autoconvolution
    c = np.convolve(x, x, mode='full')
    max_abs = np.max(np.abs(c))
    return max_abs / (s * s)

def _try_sequence(x):
    """Evaluate a candidate sequence."""
    return _compute_score(x)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Seed: all ones
    seed = np.ones(N)
    best_seq = seed.copy()
    best_score = _compute_score(seed)
    
    # Strategy: Try sequences with a bias toward +1 but with structured -1s
    # to reduce autocorrelation peaks while keeping sum large.
    
    # Approach: Use a base of +1s and flip a subset to -1 to reduce autocorrelation.
    # The sum will be N - 2*(number of -1s).
    
    # Try various fractions of -1s with different patterns
    
    # First, try simple patterns
    candidates = []
    
    # Pattern 1: Random with varying bias
    rng = np.random.RandomState(42)
    for frac in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        n_neg = int(frac * N)
        idx = rng.choice(N, n_neg, replace=False)
        x = np.ones(N)
        x[idx] = -1
        candidates.append(x)
    
    # Pattern 2: Block patterns
    for block_len in [2, 3, 4, 5, 7, 10, 15, 20]:
        for neg_frac in [0.0, 0.1, 0.2, 0.3]:
            n_neg = int(neg_frac * N)
            # Create a pattern with blocks
            pattern = np.ones(block_len)
            if neg_frac > 0:
                n_neg_in_block = max(1, int(neg_frac * block_len))
                pattern[:n_neg_in_block] = -1
            x = np.tile(pattern, N // block_len)
            if len(x) < N:
                x = np.concatenate([x, np.ones(N - len(x))])
            candidates.append(x)
    
    # Evaluate all candidates
    for x in candidates:
        if time.time() > deadline:
            break
        s = _try_sequence(x)
        if s < best_score:
            best_score = s
            best_seq = x.copy()
    
    # Local search: flip individual elements to improve
    if time.time() < deadline:
        x = best_seq.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                x[i] = -x[i]  # flip
                s = _try_sequence(x)
                if s < best_score:
                    best_score = s
                    best_seq = x.copy()
                    improved = True
                else:
                    x[i] = -x[i]  # revert
    
    # Pairwise swap search
    if time.time() < deadline:
        x = best_seq.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(0, N, 2):  # step by 2 for speed
                if time.time() > deadline:
                    break
                for j in range(i+1, min(i+20, N)):
                    if time.time() > deadline:
                        break
                    if x[i] != x[j]:
                        # Swap would change values
                        x[i], x[j] = x[j], x[i]
                        s = _try_sequence(x)
                        if s < best_score:
                            best_score = s
                            best_seq = x.copy()
                            improved = True
                        else:
                            x[i], x[j] = x[j], x[i]
    
    return best_seq
# EVOLVE-BLOCK-END
