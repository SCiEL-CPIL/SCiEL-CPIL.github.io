"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)

    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    
    # Hybrid basis: 5 Gaussian RBFs + 5 polynomial terms
    # This captures both local bumps and global trends
    # Linear in params, so BFGS converges quickly
    
    # Gaussian centers and width
    centers = np.array([-2.5, -1.25, 0.0, 1.25, 2.5])
    width = 1.5
    
    # Gaussian basis: (n, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    gauss_basis = np.exp(-0.5 * (diff / width) ** 2)
    
    # Polynomial basis: (n, 5) = [1, x, x^2, x^3, x^4]
    poly_basis = np.column_stack([
        np.ones(n),
        x,
        x**2,
        x**3,
        x**4
    ])
    
    # params[0:5] for Gaussians, params[5:10] for polynomials
    gauss_coeffs = params[0:5]
    poly_coeffs = params[5:10]
    
    pred = gauss_basis @ gauss_coeffs + poly_basis @ poly_coeffs
    
    return pred
# EVOLVE-BLOCK-END
