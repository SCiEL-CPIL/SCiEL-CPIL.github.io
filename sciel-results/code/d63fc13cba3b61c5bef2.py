import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input to [-1, 1] for numerical stability
    t = x / 3.0
    
    # Quadratic trend: params[0]=intercept, params[1]=linear, params[2]=quadratic
    y = params[0] + params[1] * t + params[2] * t * t
    
    # RBF part: 7 Gaussian RBFs with fixed centers in scaled domain
    centers = np.array([-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9])
    sigma = 0.45
    
    # Compute RBF values
    # t shape (N,), centers shape (7,)
    # We want (N, 7)
    diff = t[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel: exp(-0.5 * (diff/sigma)^2)
    # Pre-compute constant 0.5 / sigma^2
    const = 0.5 / (sigma * sigma)
    rbf_vals = np.exp(-const * diff**2)
    
    # params[3:10] are the 7 RBF amplitudes
    # rbf_vals @ params[3:10] gives (N,)
    y += rbf_vals @ params[3:10]
    
    return y
