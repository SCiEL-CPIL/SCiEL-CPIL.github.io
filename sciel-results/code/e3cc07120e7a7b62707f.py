"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# Fixed hyperparameters for the Multi-Scale RBF basis
# Domain is [0.1, 6]
# We use 8 RBF centers evenly spaced
_CENTERS = np.linspace(0.1, 6.0, 8)

# Define two scales for multi-scale RBFs
# Wide RBFs for global structure (interleaved)
_SIGMA_WIDE = 1.5
_INV_2_SIGMA2_WIDE = 1.0 / (2.0 * _SIGMA_WIDE * _SIGMA_WIDE)

# Narrow RBFs for local details (interleaved)
_SIGMA_NARROW = 0.6
_INV_2_SIGMA2_NARROW = 1.0 / (2.0 * _SIGMA_NARROW * _SIGMA_NARROW)

# Indices for wide and narrow RBFs
# Wide: indices 0, 2, 4, 6
# Narrow: indices 1, 3, 5, 7
_WIDE_INDICES = [0, 2, 4, 6]
_NARROW_INDICES = [1, 3, 5, 7]


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the model prediction using a Multi-Scale RBF basis.
    
    Model: y = p[0] + p[1]*x + sum(p[2+i] * RBF_wide(x, c_i)) + sum(p[6+i] * RBF_narrow(x, c_i))
    
    Args:
    x: Input array of shape (N,)
    params: Parameter array of shape (10,)
        - params[0]: Constant offset
        - params[1]: Linear coefficient
        - params[2:6]: Weights for 4 wide-bandwidth RBFs (centers 0-3)
        - params[6:10]: Weights for 4 narrow-bandwidth RBFs (centers 4-7)
    
    Returns:
    Prediction array of shape (N,)
    """
    # Ensure x is a 1D array for calculation
    x_flat = np.asarray(x).flatten()
    
    # params mapping
    w_const = params[0]
    w_linear = params[1]
    w_wide = params[2:6]   # 4 weights for wide RBFs
    w_narrow = params[6:10] # 4 weights for narrow RBFs
    
    # Compute diff for all centers
    # x_flat shape: (N,)
    # _CENTERS shape: (8,)
    # diff shape: (N, 8)
    diff = x_flat[:, np.newaxis] - _CENTERS[np.newaxis, :]
    
    # Extract diffs for wide and narrow centers
    diff_wide = diff[:, _WIDE_INDICES]
    diff_narrow = diff[:, _NARROW_INDICES]
    
    # Compute wide RBFs
    exp_arg_wide = -diff_wide * diff_wide * _INV_2_SIGMA2_WIDE
    rbf_wide = np.exp(exp_arg_wide) # Shape (N, 4)
    
    # Compute narrow RBFs
    exp_arg_narrow = -diff_narrow * diff_narrow * _INV_2_SIGMA2_NARROW
    rbf_narrow = np.exp(exp_arg_narrow) # Shape (N, 4)
    
    # Linear combination
    y_wide = rbf_wide @ w_wide
    y_narrow = rbf_narrow @ w_narrow
    
    # Total prediction
    y = w_const + w_linear * x_flat + y_wide + y_narrow
    
    return y
# EVOLVE-BLOCK-END
