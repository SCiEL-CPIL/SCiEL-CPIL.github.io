"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_c3_fast(h):
    """
    Compute the autoconvolution upper bound C3 for a signed step function.
    Uses FFT-based autocorrelation. Returns max absolute value of the
    autocorrelation at non-zero lags (excluding lag 0 which is just sum of squares).
    """
    n = len(h)
    # Zero-pad to avoid circular convolution
    size = 1
    while size < 2 * n:
        size *= 2
    H = np.fft.fft(h, n=size)
    ac = np.fft.ifft(H * np.conj(H)).real[:n]
    # Exclude lag 0 (self-correlation = sum of squares = N for ±1)
    # We want the max over non-zero lags
    if n > 1:
        return float(np.max(np.abs(ac[1:])))
    return 0.0

def _compute_c3_vectorized(h):
    """
    Alternative: compute C3 using direct vectorized approach for small N.
    For N=400, FFT is faster, but this is kept for verification.
    """
    n = len(h)
    # autocorr[k] = sum_{i=0}^{n-1-k} h[i]*h[i+k] for k=0..n-1
    # Use np.correlate or manual
    # Actually, let's just use the FFT version as it's faster
    return _compute_c3_fast(h)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    np.random.seed(42)
    
    best_h = None
    best_score = np.inf
    
    # Phase 0: Try a few structured initializations
    candidates = []
    
    # All ones (baseline)
    candidates.append(np.ones(N, dtype=np.float64))
    
    # Random
    candidates.append(np.random.choice([-1.0, 1.0], size=N))
    
    # Alternating
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # Block pattern: 10 ones, 10 minus ones
    block_pattern = np.tile([1.0, -1.0], N // 2)
    candidates.append(block_pattern)
    
    # Block pattern: 5 ones, 5 minus ones
    block_pattern2 = np.tile([1.0, -1.0], N // 2)  # same as above for N=400
    # Let's do 4 ones, 4 minus ones
    block_pattern3 = np.tile([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0], N // 8)
    candidates.append(block_pattern3)
    
    # Evaluate candidates
    for cand in candidates:
        score = _compute_c3_fast(cand)
        if score < best_score:
            best_score = score
            best_h = cand.copy()
    
    # Phase 1: Multi-restart simulated annealing with multi-flip moves
    rng = np.random.RandomState(42)
    
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize current solution
        if restart == 0:
            current_h = best_h.copy()
            current_score = best_score
        else:
            # Random restart
            current_h = np.random.choice([-1.0, 1.0], size=N)
            current_score = _compute_c3_fast(current_h)
        
        if current_score < best_score:
            best_score = current_score
            best_h = current_h.copy()
        
        # Simulated annealing parameters
        T = max(1.0, current_score * 0.1)
        T_min = 0.01
        cooling = 0.9999
        no_improve = 0
        max_no_improve = 20000
        
        while time.time() < deadline - 5.0:
            if no_improve > max_no_improve:
                break
            
            # Choose move type: single flip or multi-flip
            if rng.random() < 0.3:
                # Multi-flip: flip 2-4 random positions
                num_flips = rng.randint(2, 5)
                indices = rng.choice(N, size=num_flips, replace=False)
                current_h[indices] *= -1.0
                new_score = _compute_c3_fast(current_h)
                delta = new_score - current_score
                
                if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                    current_score = new_score
                    no_improve = 0
                    if new_score < best_score:
                        best_score = new_score
                        best_h = current_h.copy()
                else:
                    current_h[indices] *= -1.0  # revert
                    no_improve += 1
            else:
                # Single flip
                idx = rng.randint(N)
                current_h[idx] *= -1.0
                new_score = _compute_c3_fast(current_h)
                delta = new_score - current_score
                
                if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                    current_score = new_score
                    no_improve = 0
                    if new_score < best_score:
                        best_score = new_score
                        best_h = current_h.copy()
                else:
                    current_h[idx] *= -1.0  # revert
                    no_improve += 1
            
            T = max(T * cooling, T_min)
        
        # After SA, do a local search on the best found so far
        if time.time() < deadline - 5.0:
            local_h = best_h.copy()
            local_score = best_score
            
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for i in range(N):
                    if time.time() > deadline - 5.0:
                        break
                    local_h[i] *= -1.0
                    new_score = _compute_c3_fast(local_h)
                    if new_score < local_score:
                        local_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best_h = local_h.copy()
                    else:
                        local_h[i] *= -1.0
    
    # Phase 2: Final local search with block flips
    if time.time() < deadline - 3.0:
        final_h = best_h.copy()
        final_score = best_score
        
        for block_size in [2, 3, 4, 5, 8, 10]:
            if time.time() > deadline - 3.0:
                break
            improved = True
            while improved and time.time() < deadline - 3.0:
                improved = False
                for i in range(0, N - block_size + 1):
                    if time.time() > deadline - 3.0:
                        break
                    final_h[i:i+block_size] *= -1.0
                    new_score = _compute_c3_fast(final_h)
                    if new_score < final_score:
                        final_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best_h = final_h.copy()
                    else:
                        final_h[i:i+block_size] *= -1.0
    
    return best_h
# EVOLVE-BLOCK-END
