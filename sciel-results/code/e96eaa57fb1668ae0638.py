import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a dense Radial Basis Function (RBF) network.
    
    Strategy:
    - Place 10 Gaussian centers evenly across the input domain [-2, 2].
    - Use a fixed, relatively narrow bandwidth (sigma=0.25) to allow 
      the model to approximate local features and discontinuities.
    - The 10 parameters are the weights for these 10 Gaussian basis functions.
    - This transforms the problem into a local approximation task, 
      which is often more robust for unknown functions than global 
      polynomial or trigonometric bases.
    """
    # Define centers for the RBFs
    # Evenly spaced from -2 to 2
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Bandwidth
    sigma = 0.25
    
    # Compute the RBF values
    # x is shape (N,), centers is shape (10,)
    # diff is shape (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel
    # phi is shape (N, 10)
    phi = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    
    # Weighted sum
    # params is shape (10,)
    # prediction is shape (N,)
    prediction = phi @ params
    
    return prediction
