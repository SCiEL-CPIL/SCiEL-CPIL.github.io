import numpy as np
import time

N = 100
DEADLINE = 145.0

def _overlap_max(h):
    """Compute max cyclic overlap of h with 1-h, scaled by step width."""
    w = 2.0 / N
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = N / 2.0 - autocorr
    return w * np.max(overlaps)

def construct():
    start_time = time.time()
    
    # Theoretical lower bound for binary (0/1) sequences with sum N/2:
    # Overlap_k = N/2 - Autocorr_k.
    # For binary, Autocorr_k is integer.
    # To minimize max overlap, we want Autocorr_k as close to N/4 as possible.
    # N=100, N/4 = 25.
    # If we can find a sequence where Autocorr_k is always 25 or 24/26, 
    # max overlap is N/2 - 24 = 26 -> 0.52.
    # The trivial 0.5 solution gives overlap 0.5.
    # Wait, if h is binary, sum h_i h_{i+k} is integer.
    # Overlap = 50 - A_k.
    # If A_k = 25, Overlap = 25 -> 0.5.
    # If A_k = 24, Overlap = 26 -> 0.52.
    # If A_k = 26, Overlap = 24 -> 0.48.
    # So binary sequences can potentially beat 0.5 if they have high autocorrelation peaks (26) and low troughs (24).
    # Actually, we want to MINIMIZE the MAX overlap.
    # So we want A_k to be as LARGE as possible for the "worst" k?
    # No, Max Overlap = max_k (50 - A_k).
    # To minimize this, we want min_k A_k to be as LARGE as possible.
    # So we want the autocorrelation to be flat and high.
    # The average autocorrelation is N/4 = 25.
    # So we want A_k >= 25 for all k? Impossible if mean is 25 and variance > 0.
    # But we can have some 26s and some 24s.
    # If min A_k = 24, Max Overlap = 26 -> 0.52.
    # If min A_k = 25, Max Overlap = 25 -> 0.5.
    # If min A_k = 26, Max Overlap = 24 -> 0.48. (Requires all A_k >= 26, sum > 100, impossible).
    
    # So the best binary solution likely has max overlap 0.52 or 0.5.
    # The continuous solution 0.5 gives 0.5.
    # However, the task allows h in [0,1].
    # The seed program uses continuous h and gradient descent.
    # Let's try a hybrid: Start with binary, then allow fractional values?
    # Or just stick to binary? The prompt says "step heights h on [0,2]... bounded between 0 and 1".
    # It does NOT restrict to integers.
    
    # But the "Inspiration 1" mentioned binary patterns with SA can achieve more uniform cross-correlations.
    # Let's try a pure binary SA first. It's a structurally different search space (discrete vs continuous).
    
    # Initialize with random binary sequence with exactly N/2 ones
    h = np.zeros(N)
    ones = np.random.choice(N, N // 2, replace=False)
    h[ones] = 1.0
    
    curr_cost = _overlap_max(h)
    best_h = h.copy()
    best_cost = curr_cost
    
    T = 1.0
    T_min = 1e-4
    cooling = 0.999
    
    while time.time() < DEADLINE:
        if T < T_min:
            break
            
        # Pick a random index to flip
        i = np.random.randint(N)
        
        # Try flipping h[i]
        old_val = h[i]
        new_val = 1.0 - old_val
        
        # Check if sum constraint is violated?
        # If we flip 1->0, sum becomes N/2 - 1. Invalid.
        # If we flip 0->1, sum becomes N/2 + 1. Invalid.
        # So we must swap two elements to maintain sum = N/2.
        
        # Pick another random index j != i
        j = np.random.randint(N)
        if j == i:
            continue
            
        # Swap h[i] and h[j]
        # This maintains sum = N/2.
        # It effectively moves a "1" from i to j (if h[i]=1, h[j]=0) or vice versa.
        
        # Only make sense if h[i] != h[j]. If they are same, swap does nothing.
        if h[i] == h[j]:
            continue
            
        h[i] = 1.0 - h[i]
        h[j] = 1.0 - h[j]
        
        new_cost = _overlap_max(h)
        
        delta = new_cost - curr_cost
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            curr_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_h = h.copy()
        else:
            # Reject: revert
            h[i] = 1.0 - h[i]
            h[j] = 1.0 - h[j]
            
        T *= cooling
        
    return best_h
