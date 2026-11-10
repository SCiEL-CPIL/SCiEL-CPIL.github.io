"""Rewritten program: maximize pairwise-noncoprime subset of [1,255] containing 255.

Strategy: 255 = 3 * 5 * 17.
The condition requires gcd(a, b) > 1 for all pairs.
Any set of multiples of a single prime factor $p$ of 255 is a valid solution.
The size of such a set is floor(255 / p).
To maximize cardinality, we choose the smallest prime factor, p = 3.
Size = floor(255 / 3) = 85.
This is optimal for this specific n, as mixing different prime factors
(e.g., including a multiple of 5 not divisible by 3) forces the entire set
to be multiples of that factor (or a subset thereof), which yields a smaller
set size (51 for 5, 15 for 17).
"""
# EVOLVE-BLOCK-START
def construct():
    # 255 = 3 * 5 * 17
    # Use the smallest prime factor 3 to maximize the count of multiples.
    # All multiples of 3 in [1, 255] form a pairwise-noncoprime set containing 255.
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
