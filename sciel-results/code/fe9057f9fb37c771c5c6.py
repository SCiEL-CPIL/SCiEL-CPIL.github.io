import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model using a fixed basis of Gaussian functions.
    The model is y = sum(params[i] * Gaussian_i(x)).
    
    This approach uses a non-parametric-like basis with fixed centers and widths,
    allowing the linear coefficients (params) to adapt the height of each bump.
    This is often more robust for arbitrary functions than pure polynomials,
    especially when the optimizer starts from a fixed initialization.
    """
    # Define 10 Gaussian basis functions
    # Centers spread across [-2, 2]
    # We use 10 centers to utilize all 10 parameters
    centers = np.array([-1.8, -1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4, 1.8])
    
    # Width (sigma) of the Gaussians
    # A smaller width makes the basis more localized, a larger width makes it smoother.
    # 0.4 is a reasonable choice for a range of 4.
    sigma = 0.4
    sigma_sq = sigma * sigma
    
    # Compute the basis functions
    # x is shape (N,), centers is shape (10,)
    # We want to compute exp(-(x - c)^2 / (2 * sigma^2))
    # Using broadcasting: x[:, None] - centers[None, :]
    
    # Ensure x is 1D for safety, though evaluator likely passes 1D
    x_flat = x.flatten()
    
    # Compute squared distances
    # diff is shape (N, 10)
    diff = x_flat[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute Gaussian values
    # gaussians is shape (N, 10)
    gaussians = np.exp(-(diff ** 2) / (2.0 * sigma_sq))
    
    # Compute prediction: sum(params[i] * gaussians[:, i])
    # params is shape (10,)
    # Use dot product for efficiency
    pred = gaussians.dot(params)
    
    return pred
