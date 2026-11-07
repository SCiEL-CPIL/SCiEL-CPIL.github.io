def _chebyshev_basis(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * x - (x_min + x_max)) / (x_max - x_min)
    
    N = x.size
    B = np.empty((N, 10), dtype=np.float64)
    
    B[:, 0] = 1.0
    if N == 0:
        return B
    
    B[:, 1] = t
    
    for k in range(2, 10):
        B[:, k] = 2.0 * t * B[:, k-1] - B[:, k-2]
        
    # Normalize columns to unit variance to help BFGS
    # Calculate variance of each column
    for i in range(10):
        std = np.std(B[:, i])
        if std > 1e-10:
            B[:, i] = B[:, i] / std
            
    return B
