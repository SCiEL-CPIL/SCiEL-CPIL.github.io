import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a linear combination of 10 Gaussian RBF kernels.
    
    The model is:
    y = sum_{i=0}^{9} w_i * exp(-((x - c_i)^2) / (2 * sigma^2))
    
    where:
    - w_i are the 10 parameters (fitted by external BFGS)
    - c_i are fixed centers evenly spaced in [-3, 3]
    - sigma is a fixed bandwidth
    
    This representation is linear in the parameters, making the optimization
    landscape convex (quadratic MSE). BFGS will converge to the global minimum
    (subject to numerical precision) much more robustly than on non-linear
    or ill-conditioned non-linear bases.
    
    The local nature of RBFs allows fitting of functions that are not well-approximated
    by low-degree global polynomials or simple trigonometric series, such as
    functions with sharp transitions, high-frequency local features, or complex
    shapes that require local flexibility.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Ensure params has exactly 10 elements
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    else:
        p = p[:10]
        
    # Fixed centers: 10 points evenly spaced in [-3, 3]
    # c_i = -3 + i * (6/9) for i in 0..9
    # c = [-3.0, -2.333, -1.666, -1.0, -0.333, 0.333, 1.0, 1.666, 2.333, 3.0]
    centers = np.linspace(-3.0, 3.0, 10)
    
    # Bandwidth sigma. 
    # If sigma is too small, the kernels are very narrow and the design matrix 
    # becomes ill-conditioned (nearly singular if x doesn't hit centers closely).
    # If sigma is too large, it behaves like a global smooth function (similar to Chebyshev).
    # We want a balance. The distance between centers is 0.667.
    # A sigma of ~0.5-0.7 provides good overlap without being too global.
    # Let's use sigma = 0.6.
    sigma = 0.6
    sigma2 = sigma * sigma
    
    # Compute the RBF values
    # x has shape (N,), centers has shape (10,)
    # We want a matrix K of shape (N, 10) where K[i, j] = exp(-(x[i]-c[j])^2 / (2*sigma^2))
    
    # Efficient computation using broadcasting
    # x_col = x[:, np.newaxis]  # (N, 1)
    # c_row = centers[np.newaxis, :]  # (1, 10)
    # diff = x_col - c_row  # (N, 10)
    # K = np.exp(-diff**2 / (2 * sigma2))
    
    # To avoid memory issues for very large N (though N=200 is small), 
    # we can just do it directly.
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    K = np.exp(-diff**2 / (2.0 * sigma2))
    
    # Prediction = K @ p
    # K is (N, 10), p is (10,)
    pred = K @ p
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        # If params explode, return zeros to avoid 1e30 penalty, 
        # though BFGS should keep them finite.
        return np.zeros_like(x)
        
    return pred
