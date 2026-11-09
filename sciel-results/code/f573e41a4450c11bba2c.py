import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multi-resolution basis:
      - 4 Chebyshev polynomials (global smooth component)
      - 6 multi-scale Gaussian RBFs (local features at various widths)
    
    Parameters:
      params[0:4]   - weights for Chebyshev T_0..T_3
      params[4:10]  - weights for 6 Gaussian RBFs at different centers/scales
    
    The 6 RBFs are:
      index 4: center=-1.0, sigma=0.5
      index 5: center=0.0,  sigma=0.5
      index 6: center=1.0,  sigma=0.5
      index 7: center=-1.5, sigma=0.3  (narrow, left)
      index 8: center=0.5,  sigma=0.3  (narrow, middle-right)
      index 9: center=1.5,  sigma=0.3  (narrow, right)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
    n = x.shape[0]
    
    # Scale to [-1, 1] for Chebyshev
    t = x / 2.0
    
    basis = np.empty((n, 10), dtype=np.float64)
    
    # Chebyshev T_0 = 1
    basis[:, 0] = 1.0
    # T_1 = t
    basis[:, 1] = t
    # T_2 = 2t^2 - 1
    basis[:, 2] = 2.0 * t * t - 1.0
    # T_3 = 4t^3 - 3t
    basis[:, 3] = 4.0 * t * t * t - 3.0 * t
    
    # 6 Gaussian RBFs
    centers = np.array([-1.0, 0.0, 1.0, -1.5, 0.5, 1.5])
    sigmas  = np.array([0.5,  0.5, 0.5,  0.3,  0.3, 0.3])
    
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    gaussians = np.exp(-0.5 * (diff / sigmas[np.newaxis, :]) ** 2)
    basis[:, 4:10] = gaussians
    
    return basis @ params
