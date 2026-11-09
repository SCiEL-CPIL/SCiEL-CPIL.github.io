# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF Network with 3x3 grid of Gaussians.
    params[0] controls the inverse variance (width) of the Gaussians.
    params[1:10] are the weights for the 9 Gaussian centers.
    
    To ensure numerical stability and a good starting point for BFGS (which starts at all ones):
    We use alpha = exp(params[0] - 1.0).
    At initialization (params[0]=1), alpha = exp(0) = 1.0.
    This provides a reasonable default width for the domain [-1, 1]^2.
    
    The centers are fixed at:
    (-1, -1), (-1, 0), (-1, 1)
    ( 0, -1), ( 0, 0), ( 0, 1)
    ( 1, -1), ( 1, 0), ( 1, 1)
    """
    # Calculate alpha (inverse variance / inverse squared bandwidth)
    # Using exp(params[0] - 1.0) ensures alpha > 0 and starts at 1.0
    alpha = np.exp(params[0] - 1.0)
    
    # Extract weights
    w0 = params[1]
    w1 = params[2]
    w2 = params[3]
    w3 = params[4]
    w4 = params[5]
    w5 = params[6]
    w6 = params[7]
    w7 = params[8]
    w8 = params[9]
    
    # Precompute squared terms for efficiency
    x2 = x * x
    y2 = y * y
    
    # Calculate squared distances to each center
    # Center 0: (-1, -1)
    r2_00 = x2 + y2 + 2.0*x + 2.0*y + 2.0
    # Center 1: (-1, 0)
    r2_01 = x2 + y2 + 2.0*x + 1.0
    # Center 2: (-1, 1)
    r2_02 = x2 + y2 + 2.0*x - 2.0*y + 2.0
    # Center 3: (0, -1)
    r2_10 = x2 + y2 + 2.0*y + 1.0
    # Center 4: (0, 0)
    r2_11 = x2 + y2
    # Center 5: (0, 1)
    r2_12 = x2 + y2 - 2.0*y + 1.0
    # Center 6: (1, -1)
    r2_20 = x2 + y2 - 2.0*x + 2.0*y + 2.0
    # Center 7: (1, 0)
    r2_21 = x2 + y2 - 2.0*x + 1.0
    # Center 8: (1, 1)
    r2_22 = x2 + y2 - 2.0*x - 2.0*y + 2.0
    
    # Compute Gaussian kernels
    # We can vectorize this slightly by stacking, but explicit is often faster for small N
    g0 = np.exp(-alpha * r2_00)
    g1 = np.exp(-alpha * r2_01)
    g2 = np.exp(-alpha * r2_02)
    g3 = np.exp(-alpha * r2_10)
    g4 = np.exp(-alpha * r2_11)
    g5 = np.exp(-alpha * r2_12)
    g6 = np.exp(-alpha * r2_20)
    g7 = np.exp(-alpha * r2_21)
    g8 = np.exp(-alpha * r2_22)
    
    # Linear combination
    return (w0 * g0 + w1 * g1 + w2 * g2 + 
            w3 * g3 + w4 * g4 + w5 * g5 + 
            w6 * g6 + w7 * g7 + w8 * g8)
    # EVOLVE-BLOCK-END
