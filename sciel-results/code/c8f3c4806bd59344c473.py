import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: 9 Gaussian RBFs + 1 constant term.
    Fixed centers and widths.
    Linear in parameters.
    """
    centers = np.linspace(-2, 2, 9)
    sigma = 1.0
    
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Add constant term
    const = np.ones((len(x), 1))
    basis = np.hstack([const, basis])
    
    # params[0] is constant, params[1:10] are RBF weights
    result = basis @ params
    return result
