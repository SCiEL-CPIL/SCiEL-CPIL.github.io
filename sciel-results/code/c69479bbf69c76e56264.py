"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of 10 shifted Gaussian
    basis functions (Radial Basis Functions).
    
    The 10 parameters represent the amplitudes of 10 Gaussian kernels.
    The centers are fixed and evenly spaced across the domain [-2, 2].
    The widths are fixed to provide local resolution while maintaining smoothness.
    
    This "bump" representation allows the model to construct arbitrary smooth 
    shapes by superimposing localized peaks and troughs. This structural change 
    addresses the stagnation of the previous global basis approach by providing 
    local flexibility, which is crucial for capturing non-smooth or rapidly 
    varying features that global polynomials or trigonometric functions might 
    miss or approximate poorly (Runge's phenomenon).
    """
    start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # 10 fixed centers evenly spaced in [-2, 2]
    # Using 10 centers to match the number of parameters
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed width (sigma). 
    # A sigma of 0.6 is a reasonable choice for domain [-2, 2] with 10 points.
    # It allows some overlap between neighbors but keeps them distinct enough 
    # to allow BFGS to distinguish their amplitudes.
    sigma = 0.6
    inv_2_sigma2 = 1.0 / (2.0 * sigma * sigma)
    
    # Compute Gaussian values for all points x and all centers
    # x has shape (N,), centers has shape (10,)
    # diff has shape (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Calculate exponents
    exponents = -inv_2_sigma2 * diff * diff
    
    # Calculate Gaussians
    # exp(0)=1, exp(-large)=0. No overflow risk.
    gaussians = np.exp(exponents)
    
    # Linear combination: sum(params[i] * gaussians[:, i])
    # params has shape (10,), gaussians has shape (N, 10)
    result = gaussians @ params
    
    # Safety: replace non-finite values to avoid breaking the optimizer
    # If any value is non-finite, we return a large constant to penalize, 
    # but typically this shouldn't happen with these bounded inputs.
    if not np.all(np.isfinite(result)):
         # Return a constant large value to signal error to optimizer without crashing
         return np.full_like(result, 1e6)
    
    return result
# EVOLVE-BLOCK-END
