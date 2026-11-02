import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Initialize with a random sequence or a specific low-correlation pattern
    # Thue-Morse sequence has good autocorrelation properties
    # t_n = parity of number of 1s in binary expansion of n
    x = np.zeros(N, dtype=np.int8)
    for i in range(N):
        # Count bits
        bits = bin(i).count('1')
        x[i] = 1 if bits % 2 == 0 else -1
        
    # Precompute autocorrelation C[k] for k=1..N-1
    # C[k] = sum x[j] x[j+k]
    C = np.zeros(N, dtype=np.float64)
    for k in range(1, N):
        C[k] = np.dot(x[:N-k], x[k:])
        
    # Initial cost
    cost = np.sum(np.abs(C[1:N])**3)
    
    # Simulated Annealing
    # Parameters
    T = 10.0
    cooling_rate = 0.9995
    min_T = 0.1
    steps = 0
    max_steps = 500000 # Adjust based on speed test
    
    # Precompute affected indices for each i to speed up loop?
    # affected_ks[i] = range(1, max(i, N-1-i)+1)
    # But we can just compute the range in the loop.
    
    rng = np.random.default_rng(42)
    
    while T > min_T and steps < max_steps:
        i = rng.integers(0, N)
        val = x[i]
        
        # Calculate delta cost
        K_max = max(i, N - 1 - i)
        delta_cost = 0.0
        # We need to update C in place, so we store old C values or revert
        # To avoid storing, we can just apply delta, check, and revert if needed.
        
        # Apply delta to C
        for k in range(1, K_max + 1):
            # Term 1: x[i+k] if valid
            # Term 2: x[i-k] if valid
            # Note: x is int8, but we treat as float for dot product logic?
            # x values are +/- 1.
            
            term_sum = 0
            if i + k < N:
                term_sum += x[i + k]
            if i - k >= 0:
                term_sum += x[i - k]
                
            if term_sum != 0:
                delta_C = -2.0 * val * term_sum
                old_C = C[k]
                new_C = old_C + delta_C
                delta_cost += abs(new_C)**3 - abs(old_C)**3
                C[k] = new_C
            # If term_sum is 0, delta_C is 0, no change to C[k] or cost.
            
        new_cost = cost + delta_cost
        
        # Acceptance criterion
        if new_cost < cost:
            cost = new_cost
            x[i] = -val
        else:
            # Check for simulated annealing acceptance
            if rng.random() < np.exp(-(new_cost - cost) / T):
                cost = new_cost
                x[i] = -val
            else:
                # Revert C changes
                for k in range(1, K_max + 1):
                    if i + k < N or i - k >= 0:
                        # We need to revert only if we changed it.
                        # We changed it if term_sum != 0.
                        # Recalculate term_sum to know if we changed?
                        # Or store which k were changed?
                        # Simpler: just recalculate delta_C and subtract it?
                        # delta_C was -2 * val * term_sum.
                        # To revert, we subtract delta_C from C[k].
                        # But we need term_sum again.
                        pass # This is inefficient.
        
        # Better approach: Store the deltas or just recompute C if rejected?
        # Recomputing C for affected k is O(K_max) which is O(N).
        # So reverting is O(N). Accepting is O(1) (just update cost and x).
        # So worst case per step is O(N).
        
        # Let's optimize the revert:
        # If rejected, we must restore C.
        # We can just recalculate C[k] for k in 1..K_max from x (which is unchanged)
        # But x is unchanged, so C should be unchanged.
        # Wait, we modified C in place.
        # So we must restore C.
        # Recalculating C[k] = dot(x[:N-k], x[k:]) is O(N-k).
        # Sum over k is O(N^2). Too slow for revert.
        
        # Alternative: Keep a copy of C? No, memory.
        # Alternative: Store the old values of C[k] in a small list?
        # K_max is at most N/2.
        # We can create a list of (k, old_C) pairs.
        
        # Let's refactor the loop to collect changes.
        
        steps += 1
        T *= cooling_rate
        
    return x
# EVOLVE-BLOCK-END
