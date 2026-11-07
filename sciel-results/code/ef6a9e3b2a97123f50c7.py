"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Linear-in-Parameters Fixed-Basis RBF Network
# 
# The previous strategy allowed the external BFGS optimizer to adjust both 
# amplitudes and centers of the RBFs. This creates a highly non-convex 
# optimization landscape where BFGS often gets stuck in local minima or 
# converges slowly, leading to poor fits (score ~3397).
#
# This new strategy fixes the structural parameters (centers and widths) 
# of the RBF basis functions to a pre-determined, well-spaced grid. 
# The 10 parameters passed to `equation` are ONLY the amplitudes (weights) 
# of these fixed basis functions.
#
# Consequently, the relationship between the parameters and the output is 
# LINEAR. The external BFGS optimizer is then tasked with solving a 
# linear least-squares problem (minimizing convex MSE). BFGS is highly 
# effective and robust for such problems, typically converging to the 
# global optimum of this subspace quickly and reliably.
#
# Basis Design:
# We use 10 Radial Basis Functions (Gaussians).
# Centers c_i are uniformly distributed in [-3, 3] to cover the input domain.
# Width sigma is fixed to provide sufficient overlap for smooth interpolation.
#
# Parameters:
# p[0] ... p[9]: Amplitudes for the 10 fixed RBFs.

# Pre-compute the fixed basis structure
# Number of RBFs
N_RBF = 10
# Centers uniformly spaced in [-3, 3]
# Using linspace ensures coverage of the entire domain
CENTERS = np.linspace(-3.0, 3.0, N_RBF)
# Fixed width. 
# If centers are spaced by 6/9 = 0.66, a sigma of 0.5-0.8 allows good overlap.
# Let's use sigma = 0.6.
SIGMA = 0.6
SIGMA_SQ_2 = 2.0 * SIGMA**2

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a fixed-basis RBF network.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 amplitudes.
        
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Ensure finite params to avoid NaN propagation
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Compute the basis matrix values
    # We need y = sum_{i=0}^{9} p[i] * exp(-(x - CENTERS[i])^2 / SIGMA_SQ_2)
    
    # To be efficient and avoid loops in the inner logic if possible, 
    # though N=10 is small, we can vectorize.
    # x shape: (N,)
    # CENTERS shape: (10,)
    # diff shape: (N, 10)
    
    diff = x[:, np.newaxis] - CENTERS[np.newaxis, :]
    rbf_values = np.exp(-(diff**2) / SIGMA_SQ_2)
    
    # rbf_values shape: (N, 10)
    # p shape: (10,)
    # Result shape: (N,)
    
    result = rbf_values @ p
    
    return result
# EVOLVE-BLOCK-END
