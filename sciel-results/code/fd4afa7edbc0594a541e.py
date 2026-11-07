import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF architecture: 5 Gaussian kernels + 5 polynomial terms.
    
    5 Gaussian RBFs centered at {-3, -1.5, 0, 1.5, 3} with moderate width
    5 Polynomial terms (1, x, x^2, x^3, x^4) for global trend
    
    This localized representation handles non-periodic, smooth targets
    effectively, allowing BFGS to independently adjust local bumps.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure exactly 10 params
    if params.shape[0] != 10:
        p = np.ones(10)
        n = min(10, params.shape[0])
        p[:n] = params[:n]
        params = p
    
    # Polynomial terms: 1, x, x^2, x^3, x^4
    poly = np.column_stack([
        np.ones_like(x),
        x,
        x**2,
        x**3,
        x**4
    ])  # shape (n, 5)
    
    # Gaussian RBFs with centers at {-3, -1.5, 0, 1.5, 3}
    # Width: sigma=1.2 for moderate localization
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    sigma = 1.2
    sigma_sq = 2.0 * sigma**2
    
    # Compute Gaussian kernels: shape (5, n)
    diffs = x[None, :] - centers[:, None]  # shape (5, n)
    gaussians = np.exp(-diffs**2 / sigma_sq)  # shape (5, n)
    
    # Linear combination: 5 polynomial + 5 Gaussian = 10 terms
    y = (params[0] * poly[:, 0] +
         params[1] * poly[:, 1] +
         params[2] * poly[:, 2] +
         params[3] * poly[:, 3] +
         params[4] * poly[:, 4] +
         params[5] * gaussians[0] +
         params[6] * gaussians[1] +
         params[7] * gaussians[2] +
         params[8] * gaussians[3] +
         params[9] * gaussians[4])
    
    # Ensure finite values
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)
    
    return y
