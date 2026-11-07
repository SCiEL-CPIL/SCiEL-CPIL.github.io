"""Rewritten program: RBF basis with fixed centers for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a fixed set of Radial Basis Functions (RBFs)
    and a constant term, weighted by the 10 fitted parameters.
    
    Basis:
    - 9 Gaussian RBFs centered on a 3x3 grid within [-1, 1]x[-1, 1]
    - 1 Constant term
    """
    # 10 parameters: 9 for RBF weights, 1 for bias/constant
    # The evaluator fits these 10 params.
    
    # Define centers for the 9 RBFs
    # A 3x3 grid covers the domain [-1, 1] x [-1, 1] reasonably well
    # Centers at -0.7, 0.0, 0.7
    centers_x = np.array([-0.7, 0.0, 0.7])
    centers_y = np.array([-0.7, 0.0, 0.7])
    
    # Generate all combinations of centers
    cx, cy = np.meshgrid(centers_x, centers_y, indexing='ij')
    centers = np.column_stack((cx.ravel(), cy.ravel()))
    
    # Define the width (sigma) of the Gaussian RBFs
    # A width of ~0.5 allows overlap and smooth interpolation
    sigma = 0.5
    
    # Pre-compute the squared distances from input points (x, y) to each center
    # x and y are 1D arrays of length N (number of points)
    # We need to compute distance for each point to each of the 9 centers.
    
    # x: (N,), y: (N,)
    # centers: (9, 2)
    
    # Expand dimensions for broadcasting
    # x_diff: (N, 9)
    # y_diff: (N, 9)
    x_diff = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]
    y_diff = y[:, np.newaxis] - centers[:, 1][np.newaxis, :]
    
    # Squared Euclidean distance: (N, 9)
    dist_sq = x_diff**2 + y_diff**2
    
    # Gaussian RBF values: (N, 9)
    # exp(-dist_sq / (2 * sigma^2))
    rbf_values = np.exp(-dist_sq / (2 * sigma**2))
    
    # Extract parameters
    p = params[:9]  # Weights for the 9 RBFs
    p_const = params[9]  # Constant term
    
    # Compute the weighted sum of RBFs
    # rbf_values: (N, 9)
    # p: (9,)
    # result: (N,)
    f = rbf_values @ p + p_const
    
    return f
# EVOLVE-BLOCK-END
