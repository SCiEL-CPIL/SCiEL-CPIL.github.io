import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Returns 50 nonnegative step heights on equal intervals of [-1/4, 1/4]
    to maximize the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    
    Analysis:
    The ratio is scale-invariant.
    Let h be a step function with support width K bins.
    For a uniform distribution over K bins, the ratio scales as 1/K.
    To maximize the ratio, we must minimize K.
    The minimum non-zero support is K=1 (a single spike).
    A single spike maximizes the L2 norm relative to the L1 norm (concentration)
    and minimizes the peak of the autoconvolution relative to the mass
    (since the convolution of a spike is a spike, whereas spreading creates
    a broader convolution with lower peak density if normalized by L1).
    
    Specifically, for a spike of height H:
    ||h||_1 = H * delta
    ||h||_2^2 = H^2 * delta
    ||h*h||_inf = H^2 * delta (peak of the triangular/box convolution)
    Ratio = (H^2 * delta) / (H * delta * H^2 * delta) = 1 / delta.
    
    For a uniform distribution over K bins:
    Ratio = 1 / (K * delta).
    
    Since delta is fixed by N and the interval length, minimizing K maximizes the ratio.
    K=1 is the optimal choice.
    
    We place the spike at the center to ensure the convolution peak is well-defined
    regardless of boundary conditions (linear vs circular convolution).
    """
    
    # Create a vector of zeros
    h = [0.0] * N
    
    # Place a single unit spike at the center
    # Index 25 for N=50 (0-indexed)
    center_idx = N // 2
    h[center_idx] = 1.0
    
    return h
# EVOLVE-BLOCK-END
