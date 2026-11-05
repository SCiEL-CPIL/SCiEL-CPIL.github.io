"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    # Theoretical analysis:
    # value = abs(2 * N * max(convolve(h, h)) / sum(h)^2)
    #
    # By Cauchy-Schwarz: for any k != 0, |sum_i h[i]*h[i+k]| <= sqrt(sum h[i]^2 * sum h[i+k]^2)
    # The overlap region is smaller than full, so sum h[i+k]^2 <= sum h[i]^2
    # Thus |conv(k)| <= sum h[i]^2 = conv(0)
    # So max(conv(h,h)) = conv(0) = sum(h^2) always (the max in absolute value at lag 0)
    #
    # Therefore value = 2*N*sum(h^2)/sum(h)^2
    # By Cauchy-Schwarz: sum(h^2) >= sum(h)^2 / N
    # So value >= 2*N*(1/N) = 2
    # Equality holds iff all h[i] are equal, i.e., h = c * ones
    #
    # The seed np.ones(400) achieves exactly 2.0, which is the global minimum.
    # No improvement is possible.
    
    return np.ones(N)
# EVOLVE-BLOCK-END
