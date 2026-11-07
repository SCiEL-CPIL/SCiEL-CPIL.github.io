import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).flatten()
    p = np.asarray(params, dtype=np.float64).flatten()
    
    if not np.all(np.isfinite(p)):
        p = np.ones(10)
        
    N = len(x)
    if N == 0:
        return np.array([])
        
    # Fixed centers
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Width parameter
    sigma = 0.5
    alpha = 1.0 / (2.0 * sigma * sigma)
    
    # Compute basis functions
    # Phi[i, j] = phi_i(x_j)
    # x is (N,), centers is (10,)
    # diff is (10, N)
    diff = x[None, :] - centers[:, None]
    Phi = np.exp(-alpha * diff**2)
    
    pred = Phi @ p
    
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
