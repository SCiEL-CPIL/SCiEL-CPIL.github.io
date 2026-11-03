"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct the step heights h.
    
    Theoretical optimality proof:
    C(k) = sum_i h_i * (1 - h_{i+k})
    Sum over all shifts k: Sum_k C(k) = Sum_i h_i * (N - Sum_j h_j) = (N/2)*(N/2) = N^2/4
    Therefore average C(k) = N/4, so max_k C(k) >= N/4.
    
    Uniform h_i = 0.5 achieves C(k) = N * 0.5 * 0.5 = N/4 for every k,
    meeting the lower bound exactly. This is the unique optimal solution
    (up to measure-zero perturbations that don't change the max).
    """
    h = np.full(N, 0.5)
    # Exact sum check: N * 0.5 = 50.0 exactly in float64 for N=100
    # No adjustment needed, but verify for safety
    assert np.all(h >= 0.0) and np.all(h <= 1.0)
    assert abs(np.sum(h) - N / 2.0) < 1e-12
    return h
# EVOLVE-BLOCK-END
