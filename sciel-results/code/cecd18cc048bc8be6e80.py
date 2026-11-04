"""
Program for the suite task: Signed autocorrelation maximum absolute-value variant.

Analysis:
The objective is to minimize S = max_k |C_k| / (sum(f))^2, where C_k is the autoconvolution.
For any real vector f, C_0 = sum(f_i^2).
By Cauchy-Schwarz, (sum(f_i))^2 <= N * sum(f_i^2), so sum(f_i^2) / (sum(f_i))^2 >= 1/N.
Since max_k |C_k| >= C_0, the score S >= 1/N.
The constant vector f = [1, 1, ..., 1] achieves exactly 1/N.
Thus, the constant vector is the global optimum.
The seed program already returns this, achieving a search score of 1.0 (the maximum possible).
Any deviation will result in a score <= 1.0.
Therefore, the most efficient and correct strategy is to return the constant vector immediately.
"""

import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the optimal signed step function.
    
    The function minimizes the ratio of the maximum absolute autoconvolution 
    to the squared integral. Mathematically, this ratio is minimized by a 
    constant vector, yielding a value of 1/N.
    """
    return np.ones(N, dtype=float)
# EVOLVE-BLOCK-END
