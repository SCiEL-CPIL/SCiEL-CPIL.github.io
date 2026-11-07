"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF-based symbolic regression model with 10 learnable parameters.
    
    params layout:
      p[0]:  shared RBF width (sigma)
      p[1..p[9]]: amplitudes for 9 RBFs at fixed centers
    
    The model is:
      y = sum_{i=1}^{9} p[i] * exp(-0.5 * ((x - c_i) / p[0])^2)
    
    Centers are fixed at evenly spaced points in [-3, 3].
    The width p[0] is constrained to be positive via softplus.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure params is 1D array of length 10
    p = np.asarray(params, dtype=np.float64).flatten()
    if p.shape[0] != 10:
        # If wrong length, pad or truncate
        if p.shape[0] < 10:
            p = np.concatenate([p, np.ones(10 - p.shape[0])])
        else:
            p = p[:10]
    
    # Shared width: use softplus to ensure positivity
    sigma = np.logaddexp(0, p[0])  # softplus(p[0]) = log(1 + exp(p[0]))
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 0.01, 10.0)
    
    # Fixed centers: 9 evenly spaced points in [-3, 3]
    centers = np.linspace(-3, 3, 9)
    
    # Compute RBF terms
    # x has shape (N,), centers has shape (9,)
    # We need (N, 9) matrix of distances
    dx = x[:, np.newaxis] - centers[np.newaxis, :]  # (N, 9)
    
    # RBF values: exp(-0.5 * (dx / sigma)^2)
    rbf_vals = np.exp(-0.5 * (dx / sigma) ** 2)  # (N, 9)
    
    # Weighted sum
    y = rbf_vals @ p[1:]
    
    return y
# EVOLVE-BLOCK-END
