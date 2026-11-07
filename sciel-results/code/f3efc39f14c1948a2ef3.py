import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Radial Basis Function (RBF) model.
    
    Structural Change:
    Instead of a linear combination of fixed global basis functions (polynomials/trig),
    this model uses adaptive local basis functions (Gaussians).
    
    Parameters:
    - params[0:5] correspond to the raw centers of the 5 RBFs.
    - params[5:10] correspond to the weights of the 5 RBFs.
    
    The centers are mapped from the unconstrained parameter space to the domain [0.1, 6]
    using a sigmoid function to ensure they remain within the input range.
    This allows BFGS to adjust the location of the "bumps" to fit the target function,
    providing a non-linear optimization landscape that is distinct from linear basis models.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Split parameters into centers and weights
    raw_centers = params[0:5]
    weights = params[5:10]
    
    # Map raw centers to the domain [0.1, 6] using a sigmoid transformation.
    # sigmoid(z) = 1 / (1 + exp(-z))
    # We scale and shift to map (-inf, inf) to (0.1, 6).
    # Let s = sigmoid(z). Then c = 0.1 + 5.9 * s.
    # Using softplus or stable sigmoid to avoid overflow.
    # For simplicity and stability with BFGS, we use a direct sigmoid implementation.
    def stable_sigmoid(z):
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    
    s = stable_sigmoid(raw_centers)
    centers = 0.1 + 5.9 * s
    
    # Define a fixed width (sigma) for the Gaussians.
    # A width of ~0.5 allows local fitting.
    sigma = 0.5
    gamma = 1.0 / (2.0 * sigma**2)
    
    # Compute RBF basis functions
    # Shape of x: (N,), Shape of centers: (5,)
    # We want to compute exp(-gamma * (x[:, None] - centers[None, :])**2)
    diffs = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-gamma * diffs**2)
    
    # Linear combination with weights
    # rbf_values is (N, 5), weights is (5,)
    pred = np.sum(rbf_values * weights[np.newaxis, :], axis=1)
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
