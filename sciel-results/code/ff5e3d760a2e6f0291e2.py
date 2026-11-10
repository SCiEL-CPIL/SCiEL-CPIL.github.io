"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Linear trend + 4 Gaussian Radial Basis Functions.
    
    The 10 parameters are:
    p[0]: Linear slope (m)
    p[1]: Linear intercept (c)
    p[2], p[3]: Amplitude (a1) and width (sigma1) for RBF 1
    p[4], p[5]: Amplitude (a2) and width (sigma2) for RBF 2
    p[6], p[7]: Amplitude (a3) and width (sigma3) for RBF 3
    p[8], p[9]: Amplitude (a4) and width (sigma4) for RBF 4
    
    The RBF centers are fixed at [-1.5, -0.5, 0.5, 1.5] to ensure 
    coverage of the domain [-2, 2] and to stabilize the optimization 
    process by reducing the number of free positional parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Unpack parameters
    m = params[0]
    c = params[1]
    
    # Fixed centers for the 4 RBFs
    centers = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
    
    # Initialize result with the linear part
    result = m * x + c
    
    # Add RBF contributions
    for i in range(4):
        a_i = params[2 + 2*i]
        sigma_i = params[3 + 2*i]
        
        # Clip sigma to avoid division by zero and ensure stability
        # If sigma is too small, it becomes a delta function, which is unstable.
        # If sigma is too large, it becomes a constant.
        sigma_clipped = np.clip(sigma_i, 1e-4, 10.0)
        
        # Compute Gaussian: exp(-(x - center)^2 / (2 * sigma^2))
        # Using np.exp with negative argument is safe from overflow.
        diff = x - centers[i]
        exponent = -(diff * diff) / (2.0 * sigma_clipped * sigma_clipped)
        
        # Clip exponent to prevent underflow issues (though exp handles it, 
        # it's good practice to avoid extreme values if any)
        # Actually, exp(-large) is 0, which is fine.
        
        result += a_i * np.exp(exponent)
        
    # Safety: replace non-finite values
    result = np.where(np.isfinite(result), result, 1e6)
    
    return result
# EVOLVE-BLOCK-END
