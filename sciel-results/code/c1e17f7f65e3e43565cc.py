"""Symbolic regression: equation(x, params) with 10 free params."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using 10 parameters.
    Uses a hybrid basis: low-degree polynomials for global trends
    and Gaussian Radial Basis Functions (RBFs) for local flexibility.
    This structure is more robust than pure polynomials for functions
    with complex or localized features.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if x.ndim == 0:
        x = x.reshape(1)
        
    # Ensure finite inputs
    x = np.where(np.isfinite(x), x, 3.05)
    p = np.where(np.isfinite(p), p, 0.0)
    
    try:
        n_points = len(x)
        if n_points == 0:
            return np.zeros(0, dtype=np.float64)
            
        # 1. Global Polynomial Basis (Degree 0-2)
        # This captures the overall trend
        basis = np.zeros((n_points, 10), dtype=np.float64)
        basis[:, 0] = 1.0
        basis[:, 1] = x
        basis[:, 2] = x**2
        
        # 2. Local RBF Basis (Gaussian)
        # Centers evenly spaced in [0.1, 6]
        # We have 7 slots left (indices 3 to 9)
        centers = np.linspace(0.1, 6.0, 7)
        sigma = 0.5 # Fixed width for stability
        
        for i in range(7):
            c = centers[i]
            # Gaussian: exp(-0.5 * ((x - c) / sigma)^2)
            basis[:, 3 + i] = np.exp(-0.5 * ((x - c) / sigma)**2)
        
        # Linear combination of basis functions
        result = basis @ p
        
        # Check for non-finite values
        if not np.all(np.isfinite(result)):
            return np.zeros_like(x, dtype=np.float64)
            
        return result
    except Exception:
        return np.zeros_like(x, dtype=np.float64)
# EVOLVE-BLOCK-END
