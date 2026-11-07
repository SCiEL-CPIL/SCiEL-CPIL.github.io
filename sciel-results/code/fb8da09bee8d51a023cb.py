"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a rich basis combining polynomial,
    sinusoidal, and Gaussian terms. This allows BFGS to fit a wider class of
    hidden functions than pure polynomials alone.
    """
    x = np.asarray(x, dtype=float).flatten()
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    n_points = len(x)
    if n_points == 0:
        return np.array([])
    
    # Normalize input to [-1, 1]
    t = x / 3.0
    
    # Create basis matrix with 10 columns
    basis = np.zeros((n_points, 10))
    
    # Polynomial terms (columns 0-3)
    basis[:, 0] = 1.0          # constant
    basis[:, 1] = t            # linear
    basis[:, 2] = t ** 2       # quadratic
    basis[:, 3] = t ** 3       # cubic
    
    # Sinusoidal terms (columns 4-7)
    basis[:, 4] = np.sin(t)
    basis[:, 5] = np.cos(t)
    basis[:, 6] = np.sin(2.0 * t)
    basis[:, 7] = np.cos(2.0 * t)
    
    # Gaussian-like terms (columns 8-9)
    basis[:, 8] = np.exp(-t ** 2)
    basis[:, 9] = np.exp(-2.0 * t ** 2)
    
    # Ensure finite values
    if not np.all(np.isfinite(basis)):
        basis = np.nan_to_num(basis, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Return the linear combination of basis functions weighted by params
    return basis @ params
# EVOLVE-BLOCK-END
