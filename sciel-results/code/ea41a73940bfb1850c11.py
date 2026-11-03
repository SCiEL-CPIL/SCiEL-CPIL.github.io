"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    The objective is to minimize max|autoconv(h)| / (sum(h))^2.
    This ratio is invariant to scaling of h.
    Let h be a sequence of +1 and -1.
    autoconv(0) = sum(h^2) = N.
    The numerator is at least N.
    The denominator is (sum(h))^2.
    To minimize the ratio, we must maximize (sum(h))^2.
    The maximum possible sum for a sequence of +/- 1 is N (all +1).
    In this case, ratio = N / N^2 = 1/N.
    Any other pattern has sum < N, so (sum)^2 < N^2, leading to a larger ratio.
    Therefore, the all-ones vector is the global optimum.
    """
    return np.ones(N)
# EVOLVE-BLOCK-END
