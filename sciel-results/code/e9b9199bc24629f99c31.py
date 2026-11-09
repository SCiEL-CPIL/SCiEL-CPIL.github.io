"""
Strategy: Dense Uniform Radial Basis Function (RBF) Network.

Unlike the current program's mixed basis (polynomials, trig, sparse RBFs), which relies
on the evaluator's BFGS to navigate a potentially ill-conditioned or redundant feature
space, this program uses a dense, uniform grid of 10 Gaussian RBFs centered at evenly
spaced points in [-3, 3].

Key advantages:
1. Universal Approximation: A dense RBF network is a universal approximator. It can
   fit smooth functions of arbitrary complexity (within the resolution of the centers)
   much better than a fixed set of global polynomials or trig functions.
2. Localized Features: RBFs are localized, meaning the parameters are less correlated
   than high-degree polynomials, leading to better-conditioned optimization for BFGS.
3. Linear in Parameters: The model is linear in the params, so BFGS will converge
   reliably to the global least-squares optimum for the given basis.
"""
import numpy as np
import time
import warnings

# Suppress warnings for speed and cleanliness
warnings.filterwarnings("ignore")

# Pre-compute centers and width for the RBFs
# We have 10 parameters. We use all 10 as coefficients for 10 Gaussian RBFs.
# Centers are evenly spaced in [-3, 3].
# To ensure the RBFs cover the domain well and don't cut off too sharply at the edges,
# we can place centers at: -3, -2.33, -1.66, ..., 3.
# Or simply use np.linspace(-3, 3, 10).

N_PARAMS = 10
CENTERS = np.linspace(-3.0, 3.0, N_PARAMS)

# Width (sigma) of the Gaussians.
# If sigma is too small, the model is very flexible but parameters become highly sensitive.
# If sigma is too large, it behaves like a low-order polynomial.
# For a domain of width 6 and 10 points, spacing is 0.66.
# A sigma of roughly half the spacing (0.33) or slightly larger (0.5) is typical.
# Let's use sigma = 0.5. This provides good overlap for smooth interpolation.
SIGMA = 0.5
WIDTH_SQ = 2.0 * SIGMA**2  # 2 * sigma^2 for exp(-0.5 * ((x-c)/sigma)^2)

# Pre-compute the centers array for fast broadcasting
CENTERS_ARR = CENTERS  # shape (10,)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a linear combination of 10 Gaussian RBFs.
    
    y = sum_{i=0}^{9} params[i] * exp(-0.5 * ((x - center_i) / sigma)^2)
    
    Args:
        x: Input array of shape (N,)
        params: Array of 10 parameters (coefficients for the RBFs)
        
    Returns:
        Array of predictions of shape (N,)
    """
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Broadcast x to (N, 1) and centers to (1, 10) to compute distances
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - CENTERS_ARR[np.newaxis, :]
    
    # Compute the exponent: -0.5 * (diff / sigma)^2
    # This is equivalent to -0.5 * diff^2 / sigma^2
    # We pre-computed WIDTH_SQ = 2 * sigma^2, so -0.5 / sigma^2 = -1 / WIDTH_SQ
    # Let's just use the direct formula for clarity and speed
    # exp_term = np.exp(-0.5 * (diff / SIGMA) ** 2)
    
    # More efficient:
    # diff_sq = diff * diff
    # exp_term = np.exp(-diff_sq / WIDTH_SQ)
    
    # Calculate the RBF values
    rbf_values = np.exp(-0.5 * (diff / SIGMA) ** 2)
    
    # Linear combination: y = rbf_values @ params
    # rbf_values shape: (N, 10)
    # params shape: (10,)
    # result shape: (N,)
    y = rbf_values.dot(params)
    
    return y
