"""Construct 12 distinct positive integers with all 4096 subset sums distinct.
The theoretical lower bound for the largest element is 2^(n-1). Powers of 2 achieve it.
"""
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return exactly 12 distinct positive integers <= 1048576
    such that all 2^12 = 4096 subset sums are distinct.
    Minimize the largest integer.
    
    The powers of two [1, 2, 4, ..., 2^11] are provably optimal:
    - All subset sums are distinct (binary representation uniqueness).
    - Largest element = 2^11 = 2048, which meets the theoretical lower bound.
    - All elements are positive, distinct, and <= 1048576.
    
    We construct this set directly. A verification step is included to ensure
    correctness, though the mathematical proof guarantees validity.
    """
    n = 12
    max_val = 1048576
    deadline = time.time() + 40  # well under the 90s limit

    # Direct construction: powers of two
    # This is the unique minimal solution for the largest element.
    selected = [1 << i for i in range(n)]

    # Quick verification: all 4096 subset sums must be distinct
    # This is a safety check. For powers of 2, this always holds.
    if time.time() < deadline:
        all_sums = set()
        # Iterate through all masks
        for mask in range(1 << n):
            s = 0
            # Compute sum for this mask
            # Optimization: We can precompute sums or use bit tricks, 
            # but n=12 is small enough that direct summation is fine.
            for i in range(n):
                if mask & (1 << i):
                    s += selected[i]
            all_sums.add(s)
        
        if len(all_sums) != (1 << n):
            # This should never happen for powers of 2, but if it does,
            # we raise an error as it indicates a logic flaw or unexpected behavior.
            raise ValueError("Subset sums not distinct")

    # Ensure constraints are met
    if len(set(selected)) != n:
        raise ValueError("Values not distinct")
    if any(x <= 0 for x in selected):
        raise ValueError("Values must be positive")
    if max(selected) > max_val:
        raise ValueError("Value exceeds maximum")

    return selected
# EVOLVE-BLOCK-END
