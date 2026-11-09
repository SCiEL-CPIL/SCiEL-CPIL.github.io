"""Symbolic regression with 1 input: Chebyshev polynomial basis (degree 9)."""
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Chebyshev polynomial of degree 9 in x.
    
    y(x) = sum_{k=0}^{9} c_k * T_k(x)
    
    where T_k is the k-th Chebyshev polynomial of the first kind.
    
    This provides a well-conditioned, smooth, and expressive representation
    on [-3, 3] that can capture a wide range of nonlinear functions.
    
    Parameters:
    0..9: Chebyshev coefficients c_0 .. c_9
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure params are 1D and length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    else:
        params = params[:10]
        
    # Compute Chebyshev polynomials T_k(x) for k = 0..9
    # T_0(x) = 1
    # T_1(x) = x
    # T_{k+1}(x) = 2x*T_k(x) - T_{k-1}(x)
    
    n = len(x)
    T = np.zeros((10, n), dtype=np.float64)
    T[0, :] = 1.0
    if n > 0:
        T[1, :] = x
    
    for k in range(2, 10):
        T[k, :] = 2.0 * x * T[k-1, :] - T[k-2, :]
    
    # Linear combination
    pred = T @ params
    
    # Safety: replace non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
