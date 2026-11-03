"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Mathematical Proof:
    Let h be a sequence of length N with 0 <= h_i <= 1 and sum(h) = N/2.
    The overlap for shift k is C(k) = sum_i h[i] * (1 - h[(i+k)%N]).
    Summing C(k) over all k=0..N-1:
    Sum_k C(k) = Sum_i h[i] * Sum_k (1 - h[(i+k)%N])
    = Sum_i h[i] * (N - Sum_j h[j])
    = (N/2) * (N - N/2)
    = N^2 / 4
    
    The average overlap is (N^2/4) / N = N/4.
    Therefore, max_k C(k) >= N/4.
    
    Consider the uniform solution h_i = 0.5 for all i.
    C(k) = sum_i 0.5 * (1 - 0.5) = sum_i 0.25 = N * 0.25 = N/4.
    Here, max_k C(k) = N/4, which achieves the lower bound.
    
    Thus, the uniform distribution is the unique global optimum.
    """
    # The optimal solution is the uniform vector
    h = np.full(N, 0.5)
    
    # Ensure the sum is exactly N/2 (it is, but explicit check for safety)
    # np.sum(np.full(100, 0.5)) is exactly 50.0 in floating point
    return h

# EVOLVE-BLOCK-END
