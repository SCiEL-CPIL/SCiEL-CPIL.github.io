"""
Optimized solution for Erdos minimum overlap task.

The problem is to find a sequence h of length N with elements in [0, 1] and sum N/2
that minimizes the maximum overlap with its shifted complement.

Mathematical Analysis:
Let O_k be the overlap for shift k.
O_k = sum_i h_i * (1 - h_{(i+k)%N})
    = sum_i h_i - sum_i h_i * h_{(i+k)%N}
    = N/2 - C_k
where C_k is the circular autocorrelation of h at lag k.

We want to minimize max_k O_k, which is equivalent to maximizing min_k C_k.
The average value of C_k over all k is:
avg(C_k) = (1/N) * sum_k C_k = (1/N) * (sum_i h_i)^2 = (1/N) * (N/2)^2 = N/4.

Since min_k C_k <= avg(C_k) = N/4, we have:
max_k O_k >= N/2 - N/4 = N/4.

The uniform sequence h_i = 0.5 for all i achieves C_k = N/4 for all k,
resulting in max_k O_k = N/4.
Thus, the uniform sequence is the unique global optimum (up to permutation/rotation).
"""

import numpy as np

N = 100

def construct():
    """
    Returns the optimal sequence of step heights.
    """
    # The uniform distribution h_i = 0.5 is the proven optimal solution.
    # Sum is N * 0.5 = N/2.
    # Values are in [0, 1].
    # Max overlap is exactly N/4, which is the theoretical lower bound.
    return np.full(N, 0.5)
