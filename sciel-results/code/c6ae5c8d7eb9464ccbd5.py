import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a dense Gaussian RBF (Radial Basis Function) basis.
    
    Strategy:
    - Use 10 Gaussian RBFs centered at fixed points in [-2, 2].
    - The width of the RBFs is fixed to 0.8 to balance locality and smoothness.
    - This creates a highly flexible, non-linear basis that can approximate 
      any continuous function well, unlike the fixed polynomial/trig basis.
    - The parameters are the weights for each RBF.
    
    The fixed centers are chosen to cover the domain [-2, 2] roughly uniformly.
    """
    # Fixed centers for the RBFs, spread across [-2, 2]
    # 10 centers: -2, -1.55, -1.1, -0.65, -0.2, 0.25, 0.7, 1.15, 1.6, 2.05
    # Let's use np.linspace(-2, 2, 10) for simplicity and uniform coverage
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed width (standard deviation of the Gaussian)
    # A width of 0.8 ensures significant overlap between neighbors, 
    # allowing smooth interpolation while maintaining local adaptability.
    sigma = 0.8
    
    # Compute the RBF values: exp(-((x - c)^2) / (2 * sigma^2))
    # x shape: (N, 1) or (N,)
    # centers shape: (10,)
    # We need to broadcast to compute distance between each x_i and each c_j
    
    # Reshape x to (N, 1) and centers to (1, 10) for broadcasting
    x_col = x.reshape(-1, 1)
    c_row = centers.reshape(1, -1)
    
    # Compute squared distances
    diff = x_col - c_row
    diff_sq = diff * diff
    
    # Compute RBF values
    rbf_values = np.exp(-diff_sq / (2.0 * sigma * sigma))
    
    # rbf_values shape: (N, 10)
    # params shape: (10,)
    # We need the dot product: sum_j rbf_values[i, j] * params[j]
    prediction = rbf_values @ params
    
    return prediction
