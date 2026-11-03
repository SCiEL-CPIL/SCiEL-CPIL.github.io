"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_overlaps(h):
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    return N / 2.0 - autocorr

def construct():
    deadline = time.time() + 140
    
    # Start with a good binary initialization
    np.random.seed(42)
    h = np.zeros(N)
    indices = np.random.permutation(N)
    h[indices[:50]] = 1.0
    
    best_h = h.copy()
    best_val = np.max(_compute_overlaps(h))
    
    # Simulated annealing on binary sequences
    # Move: flip two bits (one 0->1, one 1->0) to maintain balance
    T = 1.0
    cooling = 0.999
    min_T = 0.001
    
    while time.time() < deadline and T > min_T:
        # Pick a random 1 and a random 0
        ones = np.where(h == 1)[0]
        zeros = np.where(h == 0)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
        
        i = ones[np.random.randint(len(ones))]
        j = zeros[np.random.randint(len(zeros))]
        
        # Flip: h[i] 1->0, h[j] 0->1
        h[i] = 0
        h[j] = 1
        
        new_val = np.max(_compute_overlaps(h))
        delta = new_val - best_val  # We want to minimize, so delta < 0 is good
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            # Accept
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Reject: undo
            h[i] = 1
            h[j] = 0
        
        T *= cooling
    
    # Final: ensure sum is exactly N/2
    total = np.sum(best_h)
    if abs(total - N / 2) > 1e-10:
        # This shouldn't happen for binary with 50 ones, but just in case
        best_h += (N / 2 - total) / N
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END

