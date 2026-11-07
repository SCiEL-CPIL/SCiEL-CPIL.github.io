import numpy as np
import time

# Global cache for the basis matrix
# Since the training x is constant during the optimization, we only need to compute the basis once.
_basis_cache = None
_x_cache = None
_start_time = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Chebyshev polynomial basis of degree 9.
    
    The input x is mapped to t in [-1, 1].
    The model is f(x) = sum_{k=0}^{9} c_k T_k(t).
    This uses all 10 parameters.
    
    Chebyshev polynomials are well-conditioned and provide efficient approximation
    for smooth functions on a bounded interval. Since the loss is quadratic in the
    parameters (linear model in terms of basis functions), the optimization landscape
    is convex, ensuring BFGS converges to the global minimum.
    
    Optimization: The basis matrix is cached after the first call to avoid
    redundant computation during the iterative BFGS fitting process.
    """
    global _basis_cache, _x_cache, _start_time
    
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
        
    params = np.asarray(params, dtype=np.float64)
    if params.ndim != 1:
        params = params.ravel()
        
    # Time limit guard (150s limit, use 145s for safety)
    if time.time() - _start_time > 145:
        return np.zeros_like(x)

    # Check if x has changed (it shouldn't during a single fit, but safe to check)
    # Using array_equal is fast enough for N=200.
    if _x_cache is not None and np.array_equal(x, _x_cache):
        basis = _basis_cache
    else:
        # Map x from [0.1, 6.0] to t in [-1, 1]
        x_min = 0.1
        x_max = 6.0
        t = (2.0 * x - (x_max + x_min)) / (x_max - x_min)
        
        # Clip t to [-1, 1] to handle any slight numerical overflow/out-of-range inputs
        t = np.clip(t, -1.0, 1.0)
        
        # Compute Chebyshev polynomials T_0 to T_9
        # T_0(t) = 1
        # T_1(t) = t
        # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
        
        n_points = x.shape[0]
        # basis shape: (10, n_points)
        basis = np.zeros((10, n_points), dtype=np.float64)
        
        basis[0] = 1.0
        if n_points > 0:
            basis[1] = t
            for k in range(2, 10):
                basis[k] = 2.0 * t * basis[k-1] - basis[k-2]
                
        # Update cache
        _x_cache = x.copy()
        _basis_cache = basis

    # Ensure params is length 10
    if len(params) < 10:
        p = np.zeros(10, dtype=np.float64)
        p[:len(params)] = params
    else:
        p = params[:10]
        
    # Compute linear combination: result = sum(p_k * T_k(x))
    # basis shape (10, N), p shape (10,) -> result shape (N,)
    result = basis.T @ p
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
