import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a hybrid basis of localized Gaussian bumps
    and global trigonometric terms. This structure allows for local feature capture
    (via Gaussians) and global periodic/smooth trends (via sines/cosines), which is
    more flexible than pure polynomials for arbitrary smooth functions on [0.1, 6].
    
    The 10 parameters correspond to:
    - 6 Gaussian centers (fixed) with learned amplitudes (params[0:6])
    - 2 Sine terms (fixed frequencies) with learned amplitudes (params[6:8])
    - 1 Constant offset (params[8])
    - 1 Linear term (params[9])
    
    This linear-in-parameters model ensures BFGS converges reliably to the global
    minimum for the given basis, while the basis itself provides rich representational
    power.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    if n == 0:
        return np.array([])
    
    # Fixed Gaussian centers spread across the domain [0.1, 6]
    # Using 6 centers to capture local features
    centers = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    sigma = 0.8  # Fixed width for all Gaussians
    
    # Compute Gaussian basis: 6 columns
    # G_i(x) = exp(-0.5 * ((x - c_i) / sigma)^2)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    gaussians = np.exp(-0.5 * (diff / sigma) ** 2)  # shape (n, 6)
    
    # Compute trigonometric basis: 2 columns
    # sin(2*pi*x / 6) and cos(2*pi*x / 6) - fundamental frequency over domain
    freq = 2.0 * np.pi / 6.0
    sin_term = np.sin(freq * x)  # shape (n,)
    cos_term = np.cos(freq * x)  # shape (n,)
    
    # Assemble full basis matrix (n, 10)
    # Columns: [G1, G2, G3, G4, G5, G6, sin, cos, 1, x]
    basis = np.zeros((n, 10), dtype=np.float64)
    basis[:, 0:6] = gaussians
    basis[:, 6] = sin_term
    basis[:, 7] = cos_term
    basis[:, 8] = 1.0
    basis[:, 9] = x
    
    # Linear combination
    return basis @ params
