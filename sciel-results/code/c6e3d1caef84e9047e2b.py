"""
Symbolic regression: recover an unknown function f(x) with 1 input from [-3, 3].
Strategy: Use a linear combination of 10 Gaussian Radial Basis Functions (RBFs).
Centers are evenly spaced across the domain. Widths are fixed to ensure good
overlap between neighbors, allowing the model to capture local features without
the ill-conditioning issues often associated with high-degree global polynomials.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model with 10 Gaussian RBF basis functions.
    Centers are evenly spaced in [-3, 3].
    Widths are chosen to provide sufficient overlap for smooth interpolation.
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    orig_shape = x.shape
    x_flat = x.ravel()
    
    # 10 centers evenly spaced in [-3, 3]
    # Spacing is 6.0 / 9.0
    # centers: -3, -2.333, ..., 3
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Width parameter (sigma) for the Gaussian kernels
    # A width of ~0.5-0.6 ensures that adjacent centers have significant overlap
    # (distance between centers is ~0.667).
    # Let's pick sigma = 0.6.
    sigma = 0.6
    
    # Compute the basis matrix efficiently
    # x_flat shape: (N,)
    # centers shape: (10,)
    # We want (N, 10) matrix of distances squared
    
    # To avoid creating a huge (N, 10) array if N is very large, 
    # though N=200 is small, we can do it in a loop or vectorized.
    # Vectorized:
    # diff = x_flat[:, np.newaxis] - centers[np.newaxis, :]  # (N, 10)
    # basis = np.exp(-0.5 * (diff / sigma) ** 2)             # (N, 10)
    
    # Since N is small (200), vectorization is fast and clean.
    diff = x_flat[:, np.newaxis] - centers[np.newaxis, :]
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination
    # result = basis @ p
    result = basis.dot(p)
    
    return result.reshape(orig_shape)
# EVOLVE-BLOCK-END
