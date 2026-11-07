import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Define fixed centers and width for Gaussian RBFs
    # 10 parameters correspond to 10 Gaussians
    n_params = 10
    # Centers evenly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, n_params)
    
    # Width (sigma)
    # Too small: overfitting/ill-conditioned if data is sparse? 
    # Too large: underfitting.
    # 0.4 seems like a good balance for 10 points in [-2, 2]
    sigma = 0.4 
    
    # Compute the Gaussian basis matrix
    # x is shape (N,)
    # centers is shape (10,)
    # diff is shape (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel
    # exp(-diff^2 / (2 * sigma^2))
    basis = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    
    # Linear combination
    # params is shape (10,)
    # basis is shape (N, 10)
    return basis @ params
