"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a set of distinct integers in [1, 256] such that subset sums
    are separated by cardinality.
    """
    start_time = time.time()
    deadline = start_time + 58  # Safety margin
    
    max_val = 256
    
    # We iterate N from large to small to find the maximum size.
    # Based on analysis, max N is 31.
    # We can just check from 60 down to 1.
    
    for N in range(60, 0, -1):
        if time.time() > deadline:
            break
            
        # Calculate the minimum starting value 'a' for consecutive integers
        # a, a+1, ..., a+N-1
        # Condition: a > k(N-k-1) for all k in 1..N-1
        # Max of k(N-k-1) is at k ~ (N-1)/2
        
        max_k_val = 0
        # Optimization: only check near the peak
        k_mid = (N - 1) // 2
        for k in range(max(1, k_mid - 2), min(N, k_mid + 3)):
            val = k * (N - k - 1)
            if val > max_k_val:
                max_k_val = val
        
        a_min = max_k_val + 1
        
        # Check if the set fits in [1, 256]
        if a_min + N - 1 <= max_val:
            # Construct the set
            return [int(a_min + i) for i in range(N)]
            
    # Fallback (should not happen for N=1)
    return [1]

# EVOLVE-BLOCK-END
