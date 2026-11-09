import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates a function using a linear trend and 8 Gaussian Radial Basis Functions.
    params[0]: linear slope
    params[1]: linear intercept
    params[2:10]: amplitudes for 8 RBFs
    """
    # Linear part
    y_linear = params[0] * x + params[1]
    
    # RBF part
    # Fixed centers - spread across the domain
    centers = np.linspace(-3.0, 3.0, 8)
    # Fixed width - narrower for better local resolution
    sigma = 0.5
    
    # Compute distances
    # x shape (N,), centers shape (8,)
    # We need (N, 8) for broadcasting
    x_expanded = x[:, np.newaxis]
    c_expanded = centers[np.newaxis, :]
    
    diff = x_expanded - c_expanded
    
    # Gaussian kernel
    # Using 0.5 * (diff/sigma)^2
    rbf_values = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Weighted sum of RBFs
    # rbf_values shape (N, 8)
    # params[2:10] shape (8,)
    y_rbf = rbf_values @ params[2:10]
    
    return y_linear + y_rbf
