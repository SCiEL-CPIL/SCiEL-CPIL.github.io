"""Rewritten program: maximize pairwise-noncoprime subset of [1,255] containing 255.

255 = 3 * 5 * 17. The maximum-size pairwise-noncoprime subset containing 255
is achieved by taking all multiples of the smallest prime factor, 3.
This gives floor(255/3) = 85 elements.
"""
# EVOLVE-BLOCK-START
def construct():
    # All multiples of 3 in [1, 255] form a pairwise-noncoprime set containing 255.
    # Cardinality = 85, which is optimal for this problem.
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
