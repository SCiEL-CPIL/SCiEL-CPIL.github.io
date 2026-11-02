import numpy as np
import time

N = 100
W = 2.0 / N

def _compute_overlap_max(h):
    """Compute the maximum overlap of h with shifted 1-h."""
    # autocorr[k] = sum_j h[j] h[(j+k)%N]
    # overlap[k] = W * sum_j h[j] (1 - h[(j+k)%N]) = W * (sum_j h[j] - sum_j h[j] h[(j+k)%N])
    # sum_j h[j] = N/2
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = (N / 2.0 - autocorr) * W
    return np.max(overlaps)

def _local_search(h0, max_iters, time_limit):
    """
    Local search / Simulated Annealing on discrete heights.
    Heights are restricted to {0, 0.5, 1}.
    Moves:
    1. Flip a single index (0<->1, 0.5 stays 0.5? No, 0<->0.5<->1).
    2. Swap two indices to maintain sum constraint strictly.
    """
    h = h0.copy()
    # Ensure h is on the discrete grid
    # Close to 0 -> 0, Close to 0.5 -> 0.5, Close to 1 -> 1
    # For now, we assume input is already discrete or we snap it.
    # Let's snap to nearest of {0, 0.5, 1}
    levels = np.array([0.0, 0.5, 1.0])
    h = np.array([levels[np.argmin(np.abs(levels - x))] for x in h])
    
    # Check sum. If sum != N/2, we need to fix it.
    # The problem requires sum = N/2.
    # With levels {0, 0.5, 1}, sum is a multiple of 0.5. N/2 is an integer.
    # So sum must be exactly N/2.
    # If initial h doesn't satisfy this, we should project or adjust.
    # However, the local search moves (swaps) preserve sum.
    # So we need a valid starting point.
    
    # Let's create a valid starting point if h0 is not valid.
    # A simple valid point: N/4 zeros, N/2 halfs, N/4 ones? 
    # Sum = 0*N/4 + 0.5*N/2 + 1*N/4 = N/4 + N/4 = N/2.
    # This requires N divisible by 4. N=100 is divisible by 4.
    
    sum_h = np.sum(h)
    if not np.isclose(sum_h, N / 2.0):
        # Re-initialize to a valid discrete configuration
        # Random permutation of [0]*25 + [0.5]*50 + [1]*25
        base = np.zeros(N)
        base[0:N//4] = 0.0
        base[N//4:N//4 + N//2] = 0.5
        base[N//4 + N//2:] = 1.0
        np.random.shuffle(base)
        h = base
        
    current_val = _compute_overlap_max(h)
    best_h = h.copy()
    best_val = current_val
    
    # Simulated Annealing parameters
    T = 0.05
    T_min = 0.001
    cooling_rate = 0.999
    
    start_time = time.time()
    
    for i in range(max_iters):
        if time.time() > time_limit:
            break
        
        # Choose a move
        # Move 1: Swap two random indices
        # This preserves the sum exactly.
        i1 = np.random.randint(N)
        i2 = np.random.randint(N)
        if i1 == i2:
            continue
            
        h[i1], h[i2] = h[i2], h[i1]
        
        # Evaluate new state
        # Optimization: We can compute the change in overlap max more efficiently?
        # For N=100, FFT is fast enough (~1000 ops). 
        # _compute_overlap_max is O(N log N).
        new_val = _compute_overlap_max(h)
        
        # Acceptance criterion
        if new_val < current_val or np.random.random() < np.exp((current_val - new_val) / T):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Revert
            h[i1], h[i2] = h[i2], h[i1]
            
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # We will run multiple restarts with different random seeds to explore the space.
    # Since the landscape is discrete and rugged, multiple restarts are crucial.
    
    best_h_global = None
    best_val_global = np.inf
    
    # Number of restarts depends on time available.
    # Each restart runs local search.
    # Let's estimate time per restart.
    # Local search with 5000 iterations: 5000 * (FFT cost + overhead).
    # FFT of size 100 is very fast.
    # Let's try 5000 iterations per restart.
    
    max_iters_per_restart = 5000
    num_restarts = 20  # Will adjust based on time
    
    for r in range(num_restarts):
        if time.time() > deadline - 5: # Leave 5s for final check
            break
            
        # Generate a random valid initial configuration
        # We need sum = N/2.
        # Strategy: Randomly assign 0, 0.5, 1 such that sum is N/2.
        # Let k0 be number of 0s, k1 number of 0.5s, k2 number of 1s.
        # k0 + k1 + k2 = N
        # 0.5*k1 + 1.0*k2 = N/2 => k1 + 2*k2 = N
        # From first eq: k0 = N - k1 - k2.
        # Substitute k1 = N - 2*k2:
        # k0 = N - (N - 2*k2) - k2 = k2.
        # So k0 = k2. Let k2 = k. Then k0 = k, k1 = N - 2k.
        # k can range from 0 to N/2.
        # For N=100, k in [0, 50].
        
        # Pick a random k to vary the "density" of 1s and 0s.
        # Uniform k might be good, or biased towards middle?
        # Let's just pick random k.
        k = np.random.randint(0, N // 2 + 1)
        
        h_init = np.zeros(N)
        h_init[:k] = 0.0
        h_init[k:k + (N - 2*k)] = 0.5
        h_init[k + (N - 2*k):] = 1.0
        
        np.random.shuffle(h_init)
        
        # Run local search
        h_res, val_res = _local_search(h_init, max_iters_per_restart, deadline - 2)
        
        if val_res < best_val_global:
            best_val_global = val_res
            best_h_global = h_res.copy()
            
    if best_h_global is None:
        # Fallback: Uniform
        best_h_global = np.full(N, 0.5)
        
    # Final sanity check: ensure sum is exactly N/2 and bounds are respected.
    # The local search maintains sum exactly via swaps.
    # The initial generation maintains sum exactly.
    # So sum should be exactly N/2.
    
    return best_h_global
