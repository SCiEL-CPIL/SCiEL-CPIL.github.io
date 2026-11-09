"""
Uses a Chebyshev polynomial basis combined with radial and interaction terms.
Chebyshev polynomials provide stable, near-orthogonal polynomial approximation
on the interval [-1, 1], allowing the linear-in-parameters model to capture
higher-order curvature efficiently. The radial term (x^2 + y^2) and interaction
term (xy) extend the basis to capture circular and diagonal structures.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure inputs are 1D arrays
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # Define centers for Radial Basis Functions (RBFs)
    # A 3x3 grid covering the domain [-1, 1] x [-1, 1]
    centers = np.array([
        [-1.0, -1.0],
        [ 0.0, -1.0],
        [ 1.0, -1.0],
        [-1.0,  0.0],
        [ 0.0,  0.0],
        [ 1.0,  0.0],
        [-1.0,  1.0],
        [ 0.0,  1.0],
        [ 1.0,  1.0]
    ])
    
    # Width parameter for RBFs (gamma = 1 / (2 * sigma^2))
    # gamma = 5.0 provides a balance between local flexibility and global smoothness
    gamma = 5.0
    
    # Compute RBF values efficiently
    # x and y are shape (N,)
    # centers are shape (9, 2)
    # We compute squared distances from each point to each center
    
    x_col = x[:, None]
    y_col = y[:, None]
    
    dx = x_col - centers[None, :, 0]
    dy = y_col - centers[None, :, 1]
    
    dist_sq = dx**2 + dy**2
    
    # rbf_vals shape is (N, 9)
    rbf_vals = np.exp(-gamma * dist_sq)
    
    # params shape is (10,)
    # p0..p8 are weights for the 9 RBFs
    # p9 is the weight for the constant term (bias)
    
    # Linear combination of RBFs
    # sum(rbf_vals * params[0:9], axis=1)
    rbf_contribution = np.sum(rbf_vals * params[:9], axis=1)
    
    # Add constant term
    constant_term = params[9]
    
    return rbf_contribution + constant_term
# EVOLVE-BLOCK-END
