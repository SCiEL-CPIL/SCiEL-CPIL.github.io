import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear Basis Expansion with Gaussian RBFs and a linear term.
    
    Parameters (10 total):
      - params[0]: weight for constant basis (1)
      - params[1]: weight for linear basis (x)
      - params[2:10]: weights for 8 Gaussian RBFs
    
    Basis functions:
      phi_0(x) = 1
      phi_1(x) = x
      phi_i(x) = exp(-(x - c_i)^2 / (2 * sigma^2)) for i=2..9
      
    Centers c_i are evenly spaced in [-3, 3].
    Sigma is fixed to 0.8.
    """
    w_const = params[0]
    w_lin = params[1]
    w_rbf = params[2:10]
    
    # 8 centers evenly spaced in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 8)
    sigma = 0.8
    
    # Compute RBFs
    # x shape: (N,)
    # centers shape: (8,)
    # diff shape: (N, 8)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf = np.exp(-diff**2 / (2 * sigma**2))
    
    # Linear combination
    result = w_const * np.ones_like(x) + w_lin * x + rbf @ w_rbf
    
    return result
