"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate using Radial Basis Functions (RBFs).
    
    Strategy:
    Instead of a fixed set of global basis functions (like x, x^2, sin(x)), 
    we use a set of localized Gaussian kernels centered at specific points 
    in the input domain [-3, 3].
    
    Why this is different:
    1. Local approximation: RBFs can approximate any smooth function more 
       flexibly than a low-degree polynomial or a fixed set of trig/exponential 
       functions, especially if the target has complex local variations.
    2. Conditioning: Global polynomial bases (x^5, etc.) can be ill-conditioned 
       or lead to oscillations (Runge's phenomenon). Localized kernels are 
       generally better conditioned for BFGS optimization, provided the 
       centers are well-spaced and the width is appropriate.
    3. Parameter Interpretability: The 10 parameters correspond to the 
       'amplitude' of each local bump. The optimizer can easily turn off 
       bumps that aren't needed by setting their parameters to 0.
    
    Implementation:
    - We place 10 centers evenly spaced in [-3, 3].
    - We use Gaussian kernels: K_i(x) = exp(-((x - c_i)^2) / (2 * sigma^2)).
    - We choose sigma = 0.8 to ensure significant overlap between neighbors 
      (smoothness) but enough localization to capture details.
    - The output is sum(params[i] * K_i(x)).
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Ensure params is length 10
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    else:
        p = p[:10]
        
    # Define 10 centers evenly spaced in [-3, 3]
    # Using linspace to cover the domain
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Width parameter for Gaussian kernels
    # sigma = 0.8 provides a good balance. 
    # If sigma is too small, the basis functions are too localized and 
    # optimization might struggle to coordinate them. 
    # If sigma is too large, they become almost constant and lose expressive power.
    sigma = 0.8
    
    # Compute the basis matrix
    # x shape: (N,)
    # centers shape: (10,)
    # We want a matrix of shape (N, 10) where M[i, j] = K_j(x[i])
    
    # Efficient computation using broadcasting
    # x[:, None] has shape (N, 1)
    # centers[None, :] has shape (1, 10)
    # diff has shape (N, 10)
    diff = x[:, None] - centers[None, :]
    
    # Gaussian kernel
    # K = exp(-diff^2 / (2 * sigma^2))
    # Pre-compute the constant factor
    coeff = 1.0 / (2.0 * sigma * sigma)
    
    basis = np.exp(-coeff * diff ** 2)
    
    # Prediction is the dot product of basis and params
    # basis shape: (N, 10)
    # p shape: (10,)
    pred = basis @ p
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
