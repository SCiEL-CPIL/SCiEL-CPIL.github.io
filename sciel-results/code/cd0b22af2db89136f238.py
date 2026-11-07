import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y using a small neural network structure.
    The 10 params are used as biases for the two hidden layers and the output layer.
    This allows BFGS to fit a non-linear function y = W2 * tanh(W1 * x + b1) + b2,
    where b1 and b2 are derived from params, and W1, W2 are fixed random weights
    (or deterministic pseudo-random weights) that provide the non-linearity.
    
    Actually, to make it robust and differentiable, we define a fixed architecture:
    y = sum_j tanh(a_j * x + b_j) + c * x + d
    where a_j, b_j, c, d are fixed? No, params must be fit.
    
    Let's use a simpler, highly effective non-linear basis that is smooth and captures
    local features better than global polynomials on [-3, 3]:
    A set of shifted tanh functions (like a simple neural net with fixed weights).
    
    y = sum_{i=1}^{8} p_i * tanh(x - center_i) + p_9 * x + p_10
    where center_i are fixed (e.g., evenly spaced in [-3, 3]).
    This is a linear combination of shifted tanhs, which is a standard basis for
    function approximation (similar to a 1-layer neural net with fixed input weights).
    BFGS will fit the coefficients p_i.
    """
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    if params.shape[0] != 10:
        params = params[:10] if params.shape[0] > 10 else np.pad(params, (0, 10 - params.shape[0]))
    
    # Fixed centers for the tanh basis functions
    # 8 shifted tanhs + 1 linear term + 1 constant
    centers = np.linspace(-2.5, 2.5, 8)
    
    # Compute basis
    # basis[:, 0:8] = tanh(x[:, None] - centers[None, :])
    # basis[:, 8] = x
    # basis[:, 9] = 1.0
    
    # Vectorized computation
    # x shape: (N,)
    # centers shape: (8,)
    # diff shape: (N, 8)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    basis_tanh = np.tanh(diff)
    
    basis = np.empty((x.shape[0], 10), dtype=float)
    basis[:, 0:8] = basis_tanh
    basis[:, 8] = x
    basis[:, 9] = 1.0
    
    return basis @ params
