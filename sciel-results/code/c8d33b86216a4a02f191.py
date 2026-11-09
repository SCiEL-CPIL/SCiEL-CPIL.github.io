"""Symbolic regression with Gaussian RBF universal approximator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate a Gaussian RBF model with 5 basis functions.
    
    Parameters (10 total):
    - params[0:5]: centers c_i for 5 RBFs
    - params[5:10]: widths sigma_i for 5 RBFs
    
    The amplitudes are implicitly learned by the optimizer via the interaction
    with the target. To make this work with BFGS, we use a fixed set of 5 RBFs
    where the first 5 params control centers and the next 5 control widths.
    The output is a weighted sum where weights are derived from the data fit.
    
    Actually, let's use a cleaner decomposition:
    - params[0:5]: centers
    - params[5:10]: log-widths (to keep positive)
    
    Then we need amplitudes too, but we only have 10 params.
    Let's use 3 RBFs instead:
    - params[0:3]: centers
    - params[3:6]: log-widths  
    - params[6:9]: amplitudes
    - params[9]: global offset/bias
    """
    x = np.asarray(x, dtype=np.float64)
    
    if len(x) == 0:
        return np.array([], dtype=np.float64)
    
    # Extract parameters for 3 RBFs + bias
    centers = params[0:3]
    log_widths = params[3:6]
    amplitudes = params[6:9]
    bias = params[9]
    
    # Compute widths (ensure positive via exp)
    widths = np.exp(log_widths)
    
    # Initialize result with bias
    result = np.full_like(x, bias, dtype=np.float64)
    
    # Add each RBF contribution
    for i in range(3):
        diff = x - centers[i]
        # Gaussian RBF: exp(-diff^2 / (2 * width^2))
        exponent = -diff**2 / (2.0 * widths[i]**2)
        # Clip to avoid underflow issues
        exponent = np.clip(exponent, -50, 50)
        rbf_values = np.exp(exponent)
        result = result + amplitudes[i] * rbf_values
    
    return result
# EVOLVE-BLOCK-END
