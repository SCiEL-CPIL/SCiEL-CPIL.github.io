"""
Improved construct() for distinct subset sums with minimal largest element.

For n positive integers with all 2^n subset sums distinct, the minimal
possible largest element is 2^(n-1), achieved by powers of 2: {1, 2, 4, ..., 2^(n-1)}.

Proof sketch: Sort a_1 < a_2 < ... < a_n. The set of subset sums using the first
i-1 elements has 2^(i-1) distinct values in [0, S_{i-1}] where S_{i-1} = sum(a_1..a_{i-1}).
Adding a_i, the new subset sums are {s, s+a_i : s in S_{i-1} sums}. For all 2^i sums
to be distinct, we need a_i > S_{i-1} (otherwise the two sets [0,S_{i-1}] and [a_i, a_i+S_{i-1}]
overlap). This forces a_i >= S_{i-1} + 1. Starting with a_1 = 1, we get S_{i-1} >= 2^(i-1) - 1,
so a_i >= 2^(i-1). Thus a_n >= 2^(n-1), and powers of 2 achieve equality.
"""

def construct():
    n = 12
    # Powers of 2 from 2^0 to 2^(n-1)
    return [1 << i for i in range(n)]
