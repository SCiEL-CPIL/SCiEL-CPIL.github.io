# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) model with fixed centers and polynomial trend.
    
    Structure:
    - 5 Gaussian RBFs with fixed centers at {-3, -1.5, 0, 1.5, 3}
      Amplitudes: p[0:5], Widths (log-scale): p[5:10]
    
    This linear-in-parameters structure allows BFGS to converge reliably.
    Fixed centers ensure the basis spans the entire input domain [-3, 3].
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Fixed centers
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Unpack parameters
    amplitudes = params[0:5]
    widths_log = params[5:10]
    
    # Transform widths to ensure positivity
    widths = np.exp(widths_log) + 1e-6
    
    # Compute RBF components
    # Shape: (5, N) where N is number of points
    diff = x[None, :] - centers[:, None]  # Broadcasting: (5, 1) - (1, N) -> (5, N)
    squared_diff = diff ** 2
    
    # Gaussian kernels: exp(-0.5 * (x - c)^2 / width^2)
    # Shape: (5, N)
    kernels = np.exp(-0.5 * squared_diff / (widths[:, None] ** 2))
    
    # Combine: sum over centers
    # y = sum_i amp_i * kernel_i
    y = amplitudes @ kernels
    
    return y
# EVOLVE-BLOCK-END
