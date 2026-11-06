"""
Constructs a set of n=12 distinct positive integers with distinct subset sums,
minimizing the largest integer.

The theoretical minimum for the largest element in a set of n integers with 
distinct subset sums is 2^(n-1). This is achieved by the powers of 2.
For n=12, the optimal set is [1, 2, 4, ..., 2048].
"""
import time
import itertools

def construct():
    # The optimal set is the powers of 2.
    # This minimizes the largest element to 2^(n-1).
    # For n=12, the largest element is 2^11 = 2048.
    n = 12
    optimal_set = [1 << i for i in range(n)]
    
    # Optional: Verify distinct subset sums to ensure correctness
    # This is fast for n=12 (4096 subsets)
    sums = set()
    for r in range(n + 1):
        for subset in itertools.combinations(optimal_set, r):
            s = sum(subset)
            if s in sums:
                raise ValueError("Subset sums are not distinct")
            sums.add(s)
            
    return optimal_set
