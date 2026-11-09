"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import numpy.polynomial.legendre as npleg


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate a hidden function using a combined basis of Legendre polynomials
    and trigonometric functions.
    
    The first 6 parameters correspond to Legendre polynomials P_0 through P_5.
    The remaining 4 parameters correspond to sin(x), cos(x), sin(2x), and cos(2x).
    
    Parameters:
    x : np.ndarray
        Input points.
    params : np.ndarray
        Array of 10 parameters to be fitted.
        
    Returns:
    np.ndarray
        Predicted values.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    
    if n == 0:
        return np.array([])

    # Scale x to [-1, 1] for Legendre polynomials
    u = x / 3.0
    
    # Compute Legendre polynomial values P_0(u) to P_5(u)
    # legvander returns a matrix of shape (N, 6) where column k is P_k(u)
    V_leg = npleg.legvander(u, 5)
    
    # Trigonometric terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2.0 * x)
    cos_2x = np.cos(2.0 * x)
    
    # Combine all basis functions
    # V_leg is (N, 6), we add 4 more columns
    V = np.empty((n, 10), dtype=np.float64)
    V[:, :6] = V_leg
    V[:, 6] = sin_x
    V[:, 7] = cos_x
    V[:, 8] = sin_2x
    V[:, 9] = cos_2x
    
    # Linear combination of basis functions
    return V @ params
# EVOLVE-BLOCK-END
