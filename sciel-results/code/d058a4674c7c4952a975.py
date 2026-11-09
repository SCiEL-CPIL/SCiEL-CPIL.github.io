"""
Strategy: Global Smooth Fourier (Cosine) Basis.
Instead of local piecewise linear functions which can struggle with capturing smooth global trends
or require many parameters for high fidelity, we use a global basis of cosine functions.
Cosine functions are smooth, orthogonal on symmetric intervals (conceptually), and well-conditioned
for BFGS when the frequencies are not too high. With 10 parameters, we can represent a reasonably
smooth function on [-3, 3] without the risk of Runge's phenomenon associated with high-degree polynomials
or the kinks associated with splines.

We use frequencies k = 1, 2, ..., 10.
The basis functions are cos(k * x).
Since the domain is [-3, 3], we don't need to scale x to [-1, 1] necessarily, 
but the frequencies are chosen to provide enough resolution.
k=1 covers the broad trend, k=10 covers finer details.
This is a purely linear model in the parameters, so BFGS will converge very quickly 
to the least-squares solution (or close to it) within the 120s limit.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Fourier cosine series basis.
    
    We use 10 basis functions: cos(k * x) for k = 1, 2, ..., 10.
    This allows the model to capture smooth periodic or non-periodic trends
    (via the combination of cosines) across the domain [-3, 3].
    
    The parameters correspond to the coefficients of these basis functions.
    BFGS will optimize these 10 coefficients to minimize the training MSE.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure params is 1D and has length 10
    if params.shape != (10,):
        # Safety check, though evaluator guarantees length 10
        params = params.flatten()
        if params.size != 10:
            return np.zeros_like(x)
            
    # Compute the basis functions
    # k ranges from 1 to 10
    k = np.arange(1, 11)  # shape (10,)
    
    # x has shape (N,)
    # We want to compute cos(k_i * x_j) for all i, j
    # Using broadcasting: x[:, None] * k[None, :]
    # This creates a (N, 10) matrix where element [j, i] is x_j * k_i
    
    # To be memory efficient and fast for large N (though N=200 is small),
    # we can just do:
    XK = x[:, np.newaxis] * k[np.newaxis, :]
    
    # Compute cosines
    basis = np.cos(XK)  # shape (N, 10)
    
    # Linear combination: y = basis @ params
    # basis is (N, 10), params is (10,)
    # Result is (N,)
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
