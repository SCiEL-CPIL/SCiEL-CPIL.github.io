"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score_normalized(h):
    """
    Compute the objective: max|autoconv| / (sum h)^2.
    We assume h is normalized to sum=1, so this is just max|autoconv|.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    max_ac = np.max(np.abs(ac[:size]))
    return max_ac / (s * s)

def _local_search_signs(signs, deadline=None):
    """
    Local search on sign patterns. signs is a vector of +1/-1.
    We optimize by flipping individual signs.
    """
    signs = signs.copy()
    current_val = _score_normalized(signs.astype(float))
    
    for _ in range(10):  # max passes
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            signs[i] = -signs[i]
            new_val = _score_normalized(signs.astype(float))
            if new_val < current_val:
                current_val = new_val
                improved = True
            else:
                signs[i] = -signs[i]  # revert
        if not improved:
            break
    
    return signs, current_val

def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin
    
    best_val = np.inf
    best_h = np.ones(N) / N  # Baseline
    
    # Initial baseline score
    best_val = _score_normalized(best_h)
    
    # Strategy: Use random sign patterns and local search
    # The key insight: since the problem is scale-invariant, we can work with
    # sign patterns (+1/-1) and normalize to sum=1 at the end.
    
    rng = np.random.default_rng(12345)
    
    # Generate multiple random sign patterns
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Generate random sign pattern
        signs = rng.choice([-1, 1], size=N)
        
        # Ensure sum is not zero
        s = np.sum(signs)
        if abs(s) < 1e-6:
            continue
        
        # Local search on this sign pattern
        signs_opt, val = _local_search_signs(signs, deadline=deadline - 2)
        
        if val < best_val:
            best_val = val
            # Convert back to normalized h
            best_h = signs_opt.astype(float) / np.sum(signs_opt)
    
    # Also try some structured patterns
    structured_patterns = []
    
    # Alternating
    structured_patterns.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)]))
    
    # Block patterns
    for block_size in [4, 8, 16, 32]:
        pattern = np.tile([1]*block_size + [-1]*block_size, N // (2*block_size))
        if len(pattern) < N:
            pattern = np.concatenate([pattern, np.ones(N - len(pattern))])
        structured_patterns.append(pattern[:N])
    
    # Chirp-based signs
    for freq in [1, 2, 3, 5, 8]:
        x = np.arange(N)
        phase = 2 * np.pi * freq * x**2 / (N * N)
        signs = np.sign(np.cos(phase))
        signs[signs == 0] = 1
        structured_patterns.append(signs)
    
    for sp in structured_patterns:
        if time.time() > deadline - 3:
            break
        
        s = np.sum(sp)
        if abs(s) < 1e-6:
            continue
        
        sp_opt, val = _local_search_signs(sp, deadline=deadline - 1)
        
        if val < best_val:
            best_val = val
            best_h = sp_opt.astype(float) / np.sum(sp_opt)
    
    return best_h

# EVOLVE-BLOCK-END
