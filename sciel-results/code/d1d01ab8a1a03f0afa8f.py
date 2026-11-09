import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

START_TIME = time.time()
DEADLINE = START_TIME + 140.0

# Strategy: Multi-Scale Gaussian Mixture Model (GMM) with Learnable Widths.
# The current RBF approach fixes the width (sigma) for all basis functions.
# This new approach uses 5 basis functions, each with a learnable weight and a learnable width.
# Total parameters: 5 weights + 5 widths = 10 parameters.
#
# f(x) = sum_{i=0}^{4} w_i * exp(-0.5 * ((x - c_i) / sigma_i)^2)
#
# To ensure sigma_i > 0 and differentiability, we parameterize sigma_i = exp(params[5+i]).
# The centers c_i are fixed at 5 uniform points across [0.1, 6].
#
# This is a nonlinear optimization problem because sigma_i appears in the exponent.
# However, the nonlinearity is smooth and bounded (exp is positive), making it suitable for BFGS.
# The optimizer can now adapt the "sharpness" of each bump to match the local features of the target function.
# For example, if the target function has a sharp peak near x=1.0, the optimizer can increase the weight
# and decrease the width of the basis function centered near 1.0.

DOMAIN_MIN = 0.1
DOMAIN_MAX = 6.0
N_BASIS = 5
# Fixed centers
centers = np.linspace(DOMAIN_MIN, DOMAIN_MAX, N_BASIS)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Multi-Scale Gaussian Basis.
    params: array of 10 values.
      params[0:5] are the weights w_i.
      params[5:10] are the log-widths log(sigma_i).
    """
    if len(params) != 10:
        return np.zeros_like(x)
    
    x = np.asarray(x, dtype=float)
    
    # Extract weights and log-widths
    weights = params[0:5]
    log_sigmas = params[5:10]
    
    # Compute sigmas, ensuring they are positive and not too small/large to cause numerical issues
    # Clamp log_sigmas to a reasonable range to prevent overflow/underflow in exp
    # exp(10) ~ 22000, exp(-10) ~ 0.000045. 
    # Domain is [0.1, 6], width of ~0.5 is typical. log(0.5) ~ -0.69.
    # Let's clamp log_sigmas to [-5, 5] to be safe.
    log_sigmas_clamped = np.clip(log_sigmas, -5.0, 5.0)
    sigmas = np.exp(log_sigmas_clamped)
    
    # Compute the basis functions
    # x shape: (N,)
    # centers shape: (5,)
    # sigmas shape: (5,)
    # We want a matrix B of shape (N, 5) where B[i, j] = exp(-0.5 * ((x[i] - centers[j]) / sigmas[j])^2)
    
    # Vectorized computation
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    # Normalize by sigma
    normalized_diff = diff / sigmas[np.newaxis, :]
    B = np.exp(-0.5 * normalized_diff**2)
    
    # Prediction is B @ weights
    pred = B @ weights
    return pred

# EVOLVE-BLOCK-START
# No additional logic needed. The evaluator's BFGS will fit the 10 parameters.
# The nonlinear nature (learnable widths) allows the model to adapt to local curvature.
# EVOLVE-BLOCK-END
