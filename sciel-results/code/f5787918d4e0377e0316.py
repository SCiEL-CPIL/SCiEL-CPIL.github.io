"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Radial Basis Function (RBF) network.
    
    Structure:
    - 9 RBF centers on a 3x3 grid covering [-1, 1] x [-1, 1].
    - 1 bias term.
    - Total 10 parameters (9 weights + 1 bias).
    - Fixed width sigma for all RBFs to keep the model linear in parameters,
      ensuring BFGS converges robustly to the global minimum.
    
    This approach is structurally different from the polynomial basis,
    allowing the model to capture local features and non-polynomial shapes
    more effectively, which is often beneficial for hidden functions with
    localized variations.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # 3x3 grid of centers in [-1, 1] x [-1, 1]
    centers = np.array([
        [-1.0, -1.0],
        [-1.0,  0.0],
        [-1.0,  1.0],
        [ 0.0, -1.0],
        [ 0.0,  0.0],
        [ 0.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  0.0],
        [ 1.0,  1.0]
    ])
    
    # Fixed width for RBFs
    # A width of 0.6 provides good overlap between neighboring centers
    sigma = 0.6
    
    # Compute squared distances from each point to each center
    # x and y are arrays of shape (N,)
    # centers is (9, 2)
    # We want distances of shape (N, 9)
    
    # Efficient vectorized distance calculation
    # x[:, None] is (N, 1), centers[:, 0] is (9,) -> (N, 9)
    dx = x[:, np.newaxis] - centers[:, 0][np.newaxis, :]
    dy = y[:, np.newaxis] - centers[:, 1][np.newaxis, :]
    
    dist_sq = dx**2 + dy**2
    
    # RBF values: exp(-dist_sq / (2 * sigma^2))
    rbf_vals = np.exp(-dist_sq / (2.0 * sigma**2))
    
    # rbf_vals shape is (N, 9)
    # params shape is (10,)
    # weights are params[0:9], bias is params[9]
    
    weights = params[:9]
    bias = params[9]
    
    # Linear combination: sum(rbf_vals * weights, axis=1) + bias
    predictions = rbf_vals @ weights + bias
    
    return predictions
# EVOLVE-BLOCK-END
