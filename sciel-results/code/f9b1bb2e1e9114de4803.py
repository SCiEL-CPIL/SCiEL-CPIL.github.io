"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns the optimal constant sequence.
    
    The score is defined as max(|C[k]|) / (sum(f))^2.
    Let f[i] = c for all i.
    sum(f) = N*c.
    C[0] = sum(f[i]^2) = N*c^2.
    C[k] = (N-k)*c^2 for k > 0.
    max(|C[k]|) = N*c^2 (at k=0).
    Score = (N*c^2) / (N*c)^2 = 1/N.
    
    By Cauchy-Schwarz, (sum(f))^2 <= N * sum(f^2) = N * C[0].
    So sum(f^2) / (sum(f))^2 >= 1/N.
    Since max(|C|) >= |C[0]| = sum(f^2), the score is always >= 1/N.
    The constant sequence achieves this lower bound.
    """
    return np.ones(N)
# EVOLVE-BLOCK-END
