# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = 12
    # We seek a set of n distinct positive integers with all 2^n subset sums distinct.
    # This is equivalent to a Sidon-like property for subset sums (superincreasing guarantees it).
    # The minimal largest element is achieved by the powers of 2: 1, 2, 4, ..., 2^(n-1).
    # Proof sketch: Let a_1 < a_2 < ... < a_n be the sorted sequence.
    # For distinct subset sums, a_k > sum_{i<k} a_i is sufficient.
    # To minimize a_n, we want to minimize the growth. The tightest constraint is a_k >= sum_{i<k} a_i + 1.
    # This recurrence yields a_k = 2^(k-1).
    # Any deviation (e.g., a_k > sum_{i<k} a_i + 1) would increase a_n or require a later element to be even larger.
    # Thus, the optimal solution is the powers of 2.
    # We construct this explicitly.
    
    # Using numpy for a clean vectorized construction
    exponents = np.arange(n)
    seq = (1 << exponents).tolist()
    
    # Verify the property (optional, for safety)
    # sums = set()
    # for mask in range(1 << n):
    #     s = 0
    #     for i in range(n):
    #         if mask & (1 << i):
    #             s += seq[i]
    #     if s in sums:
    #         # Collision found, but this should not happen for powers of 2
    #         pass
    #     sums.add(s)
    # if len(sums) != (1 << n):
    #     # Fallback (should never trigger)
    #     return [1 << i for i in range(n)]
    
    return seq
# EVOLVE-BLOCK-END
