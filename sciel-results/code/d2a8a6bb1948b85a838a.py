"""
Strategy: Dense Uniform Radial Basis Function (RBF) Network.

Unlike the current program's mixed basis (polynomials, trig, sparse RBFs), which relies
on the evaluator's BFGS to navigate a potentially ill-conditioned or redundant feature
space, this program uses a dense, uniform grid of 10 Gaussian RBFs centered at evenly
spaced points in [-3, 3].

Key advantages:
1. Universal Approximation: A dense RBF network is a universal approximator. It can
   fit smooth functions of arbitrary complexity (within the resolution of the centers)
   much better than a fixed set of global polynomials or trig functions.
2. Localized Features: RBFs are localized, meaning the parameters are less correlated
   than high-degree polynomials, leading to better-conditioned optimization for BFGS.
3. Linear in Parameters: The model is linear in the weights, so BFGS will converge
   quickly and reliably to the global minimum (the least-squares solution).
"""
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a dense uniform RBF basis.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters (weights for each RBF).
        
    Returns:
        Array of predictions of shape (N,).
    """
    x = np.asarray(x).ravel()
    
    # Define 10 evenly spaced centers in [-3, 3]
    # Using linspace from -3 to 3 with 10 points
    # Centers: -3, -2.333, -1.666, -1, -0.333, 0.333, 1, 1.666, 2.333, 3
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Choose a width (sigma) for the Gaussian RBFs.
    # The distance between centers is 6.0 / 9 = 0.666.
    # A sigma of roughly 0.5 to 0.7 provides good overlap for smooth interpolation.
    # Let's use sigma = 0.6.
    sigma = 0.6
    
    # Compute the RBF values: phi_i(x) = exp(-((x - c_i)^2) / (2 * sigma^2))
    # We can do this vectorized.
    # x shape: (N,)
    # centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute squared distances
    dist_sq = diff ** 2
    
    # Compute Gaussian values
    # exp(-dist_sq / (2 * sigma^2))
    inv_2_sigma_sq = 1.0 / (2.0 * sigma**2)
    rbf_values = np.exp(-dist_sq * inv_2_sigma_sq)
    
    # rbf_values shape: (N, 10)
    # params shape: (10,)
    
    # Linear combination: y = rbf_values @ params
    # This is efficient and linear in params.
    y = rbf_values.dot(params)
    
    return y
