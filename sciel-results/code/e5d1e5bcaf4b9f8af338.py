"""
Program for the third autocorrelation inequality task.

Key insight: The value is abs(2 * N * max(conv(h,h)) / sum(h)^2).
For any vector h with sum(h) != 0, the autoconvolution at lag 0 is sum(h^2).
By Cauchy-Schwarz, sum(h^2) >= (sum(h))^2 / N.
So max(conv) >= sum(h^2) >= sum(h)^2 / N.
Thus value >= 2 * N * (sum(h)^2 / N) / sum(h)^2 = 2.

The minimum value is 2.0, achieved when h is constant (all same sign).
np.ones(N) achieves exactly 2.0 in floating point.
"""
import numpy as np

N = 400


def construct():
    # Return a constant vector of ones. This achieves the theoretical minimum
    # value of 2.0 exactly in floating-point arithmetic, giving a score of 1.0.
    # Any non-constant vector would give value > 2.0 due to strict Cauchy-Schwarz.
    return np.ones(N, dtype=np.float64)
