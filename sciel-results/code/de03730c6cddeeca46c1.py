# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate an unknown function using a linear trend plus 8 Gaussian RBFs.
    params[0]: slope
    params[1]: intercept
    params[2:10]: amplitudes for 8 RBFs
    """
    # Linear part
    y_linear = params[0] * x + params[1]
    
    # RBF part: 8 Gaussian kernels with fixed centers and widths
    # Centers spread across [-3, 3]
    centers = np.linspace(-3.0, 3.0, 8)
    # Width parameter - moderate to capture local features
    sigma = 0.8
    
    # Compute RBF values: (N, 8)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Combine: y = linear + sum(amplitude_i * rbf_i)
    y_rbf = rbf_values @ params[2:10]
    
    return y_linear + y_rbf
# EVOLVE-BLOCK-END
