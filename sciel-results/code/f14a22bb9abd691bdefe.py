"""
Rewritten program to maximize the cardinality of a pairwise non-coprime subset of [1, 255] containing 255.

Strategy:
255 = 3 * 5 * 17.
The set of all multiples of the smallest prime factor (3) forms a valid clique because any two multiples of 3 share a factor of 3.
This set has size floor(255 / 3) = 85.
Any attempt to include numbers not divisible by 3 requires removing many multiples of 3 to maintain pairwise non-coprimality, resulting in a smaller total set size.
Thus, the multiples of 3 provide the maximum cardinality.
"""
# EVOLVE-BLOCK-START
def construct():
    # Return all multiples of 3 in the range [1, 255]
    return list(range(3, 256, 3))
# EVOLVE-BLOCK-END
