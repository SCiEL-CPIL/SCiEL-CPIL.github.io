import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is 1D for processing, then reshape back
    original_shape = x.shape
    x_flat = x.flatten()
    
    # Linear part
    y = params[0] * x_flat + params[1]
    
    # RBF part
    # Fixed centers
    centers = np.array([-2.25, -0.75, 0.75, 2.25])
    
    # Weights and widths from params
    weights = params[2:6]
    log_widths = params[6:10]
    
    # Ensure widths are positive and not too small to avoid division issues
    # exp is always positive.
    widths = np.exp(log_widths)
    
    # Calculate squared distances
    # x_flat: (N, 1)
    # centers: (4,) -> (1, 4)
    dist_sq = (x_flat[:, np.newaxis] - centers[np.newaxis, :]) ** 2
    
    # Calculate RBF values
    # widths: (4,) -> (1, 4)
    # dist_sq / (2 * widths^2)
    rbf_vals = np.exp(-dist_sq / (2.0 * widths[np.newaxis, :] ** 2))
    
    # Weighted sum
    y += rbf_vals @ weights
    
    return y.reshape(original_shape)
