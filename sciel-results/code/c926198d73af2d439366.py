"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian Radial Basis Function (RBF) network with 5 kernels.
    
    First 5 params: centers of Gaussian kernels on [-2, 2]
    Last 5 params: amplitudes (weights) for each kernel
    
    This local, flexible representation can capture sharp features,
    discontinuities, and non-smooth behavior that global polynomials
    struggle with.
    """
    # Split params into centers and amplitudes
    centers = params[:5]  # 5 kernel centers
    amplitudes = params[5:]  # 5 kernel amplitudes
    
    # Fixed width for Gaussian kernels
    # Width of 0.5 gives good local coverage on [-2, 2]
    width = 0.5
    inv_2w2 = 1.0 / (2.0 * width * width)
    
    # Compute Gaussian RBFs: phi_i(x) = exp(-(x - c_i)^2 / (2*w^2))
    # x shape: (n_points,), centers shape: (5,)
    # diff shape: (n_points, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute exponentials
    exponents = -diff * diff * inv_2w2
    basis = np.exp(exponents)  # shape: (n_points, 5)
    
    # Linear combination with amplitudes
    result = basis @ amplitudes
    
    return result
# EVOLVE-BLOCK-END
