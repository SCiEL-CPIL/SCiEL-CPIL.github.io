"""
Single-hidden-layer neural network (MLP) with 10 parameters.
The hidden layer uses 10 Gaussian radial basis functions (RBFs) centered
evenly across the input domain [0.1, 6] with a fixed width. The 10
parameters are the output weights that multiply the RBF activations.

This is a structural change from a linear polynomial basis: instead of
global polynomial basis functions, the model uses local Gaussian kernels
whose weighted sum can approximate arbitrary smooth non-linear shapes.
Because the model is linear in params, BFGS will converge to the global
optimum for the fixed centers and width.
"""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    MLP with a single hidden layer of 10 Gaussian RBFs.
    
    params: array of length 10, interpreted as the output weights.
    
    1. Define 10 RBF centers evenly spaced in [0.1, 6].
    2. Fixed Gaussian width chosen to balance locality and coverage.
    3. Compute RBF activations: phi_k(x) = exp(-((x - c_k)^2) / (2 * sigma^2)).
    4. Prediction = sum_k params[k] * phi_k(x).
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure x is a 1-D array
    x_flat = x.ravel()
    
    # 10 evenly spaced centers in [0.1, 6]
    x_min = 0.1
    x_max = 6.0
    centers = np.linspace(x_min, x_max, 10)
    
    # Fixed width for the Gaussian RBFs.
    # The spacing between adjacent centers is (x_max - x_min) / 9 = 5.9 / 9 ≈ 0.656.
    # Choose sigma so that adjacent RBFs have moderate overlap.
    # sigma = spacing / 2 gives reasonable coverage.
    spacing = (x_max - x_min) / 9.0
    sigma = spacing / 2.0
    sigma_sq = 2.0 * sigma ** 2
    
    # Compute RBF values: shape (n_points, 10)
    diff = x_flat[:, np.newaxis] - centers[np.newaxis, :]  # (n, 10)
    rbf_values = np.exp(-(diff ** 2) / sigma_sq)
    
    # Output: weighted sum of RBF activations
    prediction = rbf_values @ params[:10]
    
    return prediction
