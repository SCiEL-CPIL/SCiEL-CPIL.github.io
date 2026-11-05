"""
Constructs a set of 12 distinct positive integers with distinct subset sums,
minimizing the largest integer.

The theoretical minimum for the largest integer in a set of n distinct positive
integers with distinct subset sums is 2^(n-1). This is achieved by the powers
of 2: 1, 2, 4, ..., 2^(n-1).

For n=12, the optimal largest integer is 2048.
"""

def construct():
    n = 12
    # The optimal solution is the powers of 2
    return [1 << i for i in range(n)]
