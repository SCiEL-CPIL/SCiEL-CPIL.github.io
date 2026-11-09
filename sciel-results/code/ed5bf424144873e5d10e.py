import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Linear trend + 8 Gaussian RBFs.
    RBF centers are placed at Chebyshev nodes for better distribution.
    Width is fixed at 1.0 for smoother approximation.
    """
    # Linear part
    y_linear = params[0] * x + params[1]
    
    # RBF part
    # Chebyshev nodes for 8 centers in [-1, 1]
    k = np.arange(8)
    # Nodes in [-1, 1]
    t_centers = np.cos(np.pi * (k + 0.5) / 8.0)
    # Map to [-3, 3]
    centers = 3.0 * t_centers
    
    # Sort centers for determinism (cos is descending)
    centers = np.sort(centers)
    
    sigma = 1.0
    
    # Compute RBF values
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Amplitudes are params[2:10]
    y_rbf = rbf_values @ params[2:10]
    
    return y_linear + y_rbf
