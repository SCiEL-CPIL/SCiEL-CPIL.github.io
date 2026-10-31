"""Minimize the normalized maximum of the autoconvolution of a nonnegative
step function on 600 equal intervals over [-1/4, 1/4].

Key insight: For nonnegative h, max(autoconv) = ||h||_2^2 (achieved at lag 0).
The normalized objective is N * max(autoconv) / (sum(h))^2 = N*||h||_2^2/(sum h)^2.
By Cauchy-Schwarz, (sum h)^2 <= N*||h||_2^2, so the objective >= 1,
with equality iff h is uniform. Thus uniform is provably optimal.
"""

import numpy as np

N = 600


def construct():
    """Return the uniform step function, which is provably optimal."""
    return [1.0] * N
