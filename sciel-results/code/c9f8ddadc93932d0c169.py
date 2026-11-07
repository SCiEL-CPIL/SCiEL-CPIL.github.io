import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Radial Basis Function (RBF) network with 4 Gaussian kernels.
    
    Model: y(x) = B + A * sum_i exp(-(x - c_i)^2 / (2 * sigma_i^2))
    
    Parameters:
    - params[0:4]: Scaled center positions (mapped via tanh to [-2, 2] with offsets to break symmetry)
    - params[4:8]: Log-widths (mapped via exp to positive widths)
    - params[8]: Global amplitude A
    - params[9]: Global offset B
    
    This structural change from global polynomials to local kernels allows the model 
    to better fit functions with localized features or multiple peaks.
    """
    # Unpack parameters
    c_params = params[0:4]
    w_params = params[4:8]
    A = params[8]
    B = params[9]
    
    # Define offsets to spread initial centers when params start at 1.0
    # This helps BFGS escape the symmetry of identical initial centers.
    offsets = np.array([-1.5, -0.5, 0.5, 1.5])
    
    # Compute centers: map to [-2, 2]
    centers = 2.0 * np.tanh(c_params + offsets)
    
    # Compute widths: ensure positive
    # Start with exp(1) ~ 2.7, which is wide, allowing the optimizer to narrow them.
    widths = 0.1 + np.exp(w_params)
    
    # Compute Gaussian kernels
    # x shape: (N,), centers shape: (4,)
    # We want a (N, 4) matrix of distances
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute exponents
    # Avoid division by zero if widths are tiny (though 0.1 floor prevents this)
    exponents = -0.5 * (diff ** 2) / (widths[np.newaxis, :] ** 2)
    
    # Compute sum of Gaussians
    gaussians = np.exp(exponents)
    y_sum = np.sum(gaussians, axis=1)
    
    # Apply global amplitude and offset
    y = B + A * y_sum
    
    return y
