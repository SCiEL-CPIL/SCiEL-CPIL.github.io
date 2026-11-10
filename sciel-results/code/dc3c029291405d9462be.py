"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 shifted Gaussian basis functions.
    
    The 10 parameters represent the amplitudes of 10 Gaussian kernels.
    The centers are fixed and evenly spaced across the domain [-2, 2].
    The widths are fixed to provide local resolution while maintaining smoothness.
    
    This "bump" representation allows the model to construct arbitrary smooth 
    shapes by superimposing localized peaks and troughs.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # 10 fixed centers evenly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed width (sigma). 
    # A sigma of 0.6 is a reasonable choice for domain [-2, 2] with 10 points.
    # It allows some overlap between neighbors but keeps them distinct.
    sigma = 0.6
    inv_2_sigma2 = 1.0 / (2.0 * sigma * sigma)
    
    # Compute Gaussian values for all points x and all centers
    # x has shape (N,), centers has shape (10,)
    # diff has shape (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Calculate exponents
    exponents = -inv_2_sigma2 * diff * diff
    
    # Clip exponents to prevent underflow to 0 (which is fine) or overflow (impossible here as max is 0)
    # Actually, exp(0)=1, exp(-large)=0. No overflow risk.
    gaussians = np.exp(exponents)
    
    # Linear combination: sum(params[i] * gaussians[:, i])
    # params has shape (10,), gaussians has shape (N, 10)
    result = gaussians @ params
    
    # Safety: replace non-finite values
    result = np.where(np.isfinite(result), result, 1e6)
    
    return result
# EVOLVE-BLOCK-END
