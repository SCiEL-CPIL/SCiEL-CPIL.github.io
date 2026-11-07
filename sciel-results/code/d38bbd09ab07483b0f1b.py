import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Multi-Scale Gaussian Radial Basis Function (RBF) network.
    
    Strategy:
    - The 10 parameters are split into 5 centers (c_i) and 5 amplitudes (a_i).
    - The widths (sigma_i) are fixed to a multi-scale set {0.1, 0.2, 0.5, 1.0, 2.0}.
    - This allows the model to localize and fit sharp features or discontinuities
      that global polynomial or trigonometric bases might smooth over.
    - The model is: f(x) = sum_{i=1}^{5} a_i * exp(-((x - c_i)^2) / (2 * sigma_i^2))
    """
    # Deadline guard
    start_time = time.time()
    
    # Ensure x is 1D
    x = np.asarray(x).flatten()
    
    # Check for non-finite input
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)

    # Split params into centers and amplitudes
    # params: [c1, c2, c3, c4, c5, a1, a2, a3, a4, a5]
    centers = params[:5]
    amplitudes = params[5:]
    
    # Fixed multi-scale widths
    sigmas = np.array([0.1, 0.2, 0.5, 1.0, 2.0])
    
    # Initialize result
    result = np.zeros_like(x)
    
    # Compute RBF contributions
    for i in range(5):
        c = centers[i]
        a = amplitudes[i]
        s = sigmas[i]
        
        # Avoid division by zero or underflow issues, though sigmas are fixed and safe
        # Gaussian: exp(-0.5 * ((x - c) / s)^2)
        z = (x - c) / s
        # Clip z to prevent overflow in exp, though for Gaussian it's exp(-positive) so it underflows to 0
        # Underflow is safe (becomes 0), overflow is not possible for -0.5*z^2
        gauss = np.exp(-0.5 * z * z)
        result += a * gauss
        
        # Time check
        if time.time() - start_time > 110.0:
            break

    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
