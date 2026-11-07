import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params is a flat array of length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
        
    # Fixed centers for RBFs
    # Using 10 centers evenly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Width of the Gaussian kernels
    # 0.5 allows for some overlap, capturing local features while maintaining global smoothness
    sigma = 0.5
    
    # Compute the RBF basis matrix
    # x shape: (N,)
    # centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel: exp(-0.5 * (diff/sigma)^2)
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination of basis functions
    # basis shape: (N, 10)
    # params shape: (10,)
    return basis @ params
