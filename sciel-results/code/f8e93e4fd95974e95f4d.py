"""
Strategy: Uniform Radial Basis Function (RBF) Basis with Fixed Centers.

The current program uses a mix of global (polynomials, trigonometric) and
local (RBFs) basis functions, but the RBFs are irregularly spaced and do not
uniformly cover the input domain [-3, 3]. This can lead to poor approximation
in the tails of the domain or ill-conditioning due to correlation between
polynomial and RBF terms.

This new strategy uses a uniform grid of 9 RBFs centered at evenly spaced
points in [-3, 3], plus a constant term. This creates a "RBF Network" with
fixed centers and trainable weights. The model is linear in the parameters,
ensuring that the evaluator's BFGS optimizer will converge to the global
optimum for this basis. The uniform coverage ensures consistent approximation
quality across the entire input range.
"""
# EVOLVE-BLOCK-START
import numpy as np

# Pre-compute the RBF centers and width
# We use 9 RBFs and 1 constant term, totaling 10 parameters.
# Centers are evenly spaced in [-3, 3]
_RBF_CENTERS = np.linspace(-3.0, 3.0, 9)
# Width (sigma) of the RBFs. 
# A smaller sigma makes them more local; larger makes them more global.
# With 9 centers over a range of 6, a spacing of 0.75.
# Sigma = 0.5 * spacing is a common choice for good overlap without too much correlation.
_RBF_SIGMA = 0.5 * (6.0 / 8.0)  # 0.375
# Pre-compute the inverse of 2*sigma^2 for efficiency
_INV_2_SIGMA2 = 1.0 / (2.0 * _RBF_SIGMA**2)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a uniform RBF basis.
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters. 
                params[0:9] are weights for the 9 RBFs.
                params[9] is the constant bias.
    
    Returns:
        Array of predictions of shape (N,).
    """
    x = np.asarray(x).ravel()
    
    # Compute the 9 RBF values
    # x shape: (N,)
    # centers shape: (9,)
    # We want (N, 9) matrix of RBF values
    
    # Efficient computation:
    # diff = x[:, None] - centers[None, :]  -> (N, 9)
    # rbf_vals = exp(-diff**2 * inv_2_sigma2)
    
    diff = x[:, None] - _RBF_CENTERS[None, :]
    rbf_vals = np.exp(-diff**2 * _INV_2_SIGMA2)
    
    # rbf_vals shape: (N, 9)
    # params[0:9] shape: (9,)
    
    # Compute the linear combination of RBFs
    # y_rbf = rbf_vals @ params[0:9]
    y_rbf = rbf_vals.dot(params[:9])
    
    # Add the constant bias
    y_bias = params[9]
    
    return y_rbf + y_bias

# EVOLVE-BLOCK-END
