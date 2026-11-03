import numpy as np
import time

N = 100

def construct():
    """
    Construct step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Theoretical proof of optimality:
    - For any h with sum N/2 and 0 <= h_i <= 1:
      C(k) = sum_i h[i] * (1 - h[(i+k)%N]) for circular shift k
      Sum over all k: sum_k C(k) = sum_k sum_i h[i]*(1-h[(i+k)%N])
                           = sum_i h[i] * (N - sum_j h[j])
                           = (N/2) * (N - N/2) = N^2/4
      Average C(k) = N^2/4 / N = N/4
      Therefore max_k C(k) >= N/4
    - Uniform h_i = 0.5 gives C(k) = sum_i 0.5*0.5 = N/4 for ALL k
    - So uniform achieves the lower bound and is the unique global optimum (up to permutation)
    
    We return the exact uniform solution with guaranteed numerical precision.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # The provably optimal solution: uniform h_i = 0.5
    # Using exact construction to avoid floating-point accumulation errors
    h = np.full(N, 0.5)
    
    # Verify sum is exactly N/2 (should be exact for 0.5 * 100)
    total = float(np.sum(h))
    target = N / 2.0
    
    if total != target:
        # Correct any tiny floating-point discrepancy
        # Distribute the correction minimally
        diff = target - total
        h[0] += diff
        # Ensure bounds are respected
        h = np.clip(h, 0.0, 1.0)
        # Re-verify after clipping
        total2 = float(np.sum(h))
        if total2 != target:
            # Fine-grained correction: adjust multiple elements
            diff = target - total2
            per_elem = diff / N
            h += per_elem
            h = np.clip(h, 0.0, 1.0)
    
    # Final verification
    final_total = float(np.sum(h))
    if final_total != target:
        # Last resort: exact reconstruction
        h = np.full(N, 0.5)
    
    return h
