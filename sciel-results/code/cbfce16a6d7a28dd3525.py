"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_overlaps(h):
    """Compute overlap at all shifts: sum_i h[i]*(1-h[(i+k)%N]) for each k."""
    N = len(h)
    # overlap(k) = sum_i h[i] * (1 - h[(i+k) % N])
    # = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    # = S - autocorr_h(k) where S = sum(h) = N/2
    # autocorr_h(k) = sum_i h[i]*h[(i+k)%N]
    # For binary: autocorr_h(k) = number of positions where both are 1
    ones = np.where(h > 0.5)[0]
    if len(ones) == 0:
        return np.zeros(N)
    # Use FFT for cross-correlation
    # autocorr(k) = sum_i h[i]*h[(i+k)%N]
    # This is circular cross-correlation of h with itself
    H = np.fft.rfft(h, n=N)
    ac = np.fft.irfft(H * np.conj(H), n=N)[:N]
    # ac[k] = sum_i h[i]*h[(i-k)%N] ... need to be careful
    # Actually irfft(FFT(h)*conj(FFT(h))) gives circular autocorrelation
    # ac[k] = sum_i h[i+k]*h[i] = sum_i h[i]*h[i+k] (circular)
    # So overlap(k) = S - ac[k]
    S = h.sum()
    overlaps = S - ac
    return overlaps

def _max_overlap(h):
    """Return the maximum overlap value (not normalized)."""
    ov = _compute_overlaps(h)
    return np.max(ov)

def _max_overlap_normalized(h):
    """Return C5 = max_overlap / N."""
    return _max_overlap(h) / N

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with a random balanced binary sequence
    np.random.seed(42)
    
    # Create initial sequence: N/2 ones and N/2 zeros, randomly placed
    idx = np.random.choice(N, N//2, replace=False)
    h = np.zeros(N)
    h[idx] = 1.0
    
    best_h = h.copy()
    best_score = _max_overlap_normalized(h)
    
    # Simulated annealing / local search
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() < deadline:
        # Try a random swap: pick one 1 and one 0, swap them
        ones = np.where(h > 0.5)[0]
        zeros = np.where(h <= 0.5)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
        
        i = ones[np.random.randint(len(ones))]
        j = zeros[np.random.randint(len(zeros))]
        
        if i == j:
            continue
        
        # Make the swap
        h[i] = 0.0
        h[j] = 1.0
        
        new_score = _max_overlap_normalized(h)
        delta = new_score - best_score
        
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
            best_h = h.copy()
            best_score = new_score
        else:
            # Revert
            h[i] = 1.0
            h[j] = 0.0
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final polish: try all single swaps to see if we can improve
    h = best_h.copy()
    improved = True
    while improved:
        improved = False
        ones = np.where(h > 0.5)[0]
        zeros = np.where(h <= 0.5)[0]
        for i in ones:
            for j in zeros:
                if i == j:
                    continue
                h[i] = 0.0
                h[j] = 1.0
                new_score = _max_overlap_normalized(h)
                if new_score < best_score:
                    best_h = h.copy()
                    best_score = new_score
                    improved = True
                else:
                    h[i] = 1.0
                    h[j] = 0.0
                if time.time() > deadline - 2.0:
                    break
            if time.time() > deadline - 2.0:
                break
    
    return best_h
# EVOLVE-BLOCK-END
