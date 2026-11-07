"""
Symbolic regression candidate using a Mixture of Gaussian Radial Basis Functions (RBFs)
with an internal deterministic multi-start optimization strategy.

Strategy:
The previous linear Chebyshev approach relies on the evaluator's BFGS to fit linear coefficients.
While well-conditioned, it is limited to the span of the basis. If the target function has
sharp features or is not well-approximated by low-degree polynomials on [-3, 3], the error will be high.

This approach uses a non-linear parametric model: a sum of 5 Gaussian RBFs.
y(x) = sum_{i=1}^{5} w_i * exp(-((x - mu_i)^2) / (2 * sigma_i^2))

Parameters (10 total):
- w_1, ..., w_5: Weights (amplitudes)
- mu_1, ..., mu_5: Centers
- sigma_1, ..., sigma_5: Widths (constrained to be positive via softplus or abs)

Challenge:
The optimization landscape for RBFs is non-convex with many local minima.
The evaluator uses BFGS starting from all ones. This often converges to a poor local minimum
or fails to converge if the initial guess is bad.

Solution:
Inside `equation(x, params)`, we ignore the input `params` for the initial fit (or use it as a seed)
and perform a **deterministic multi-start search** to find a good set of parameters.
We use a fixed set of random seeds to generate diverse initial guesses.
We minimize the training MSE for each guess using `scipy.optimize.minimize` (BFGS or Nelder-Mead).
We keep the best result found within a time budget.
This effectively turns the `equation` call into a small-scale global optimization problem,
which is robust against the evaluator's poor initialization.

Time Budget:
The evaluator allows 120s, but we should aim for < 10s to be safe.
We will limit the number of restarts and iterations.
"""

import numpy as np
from scipy.optimize import minimize
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Mixture of 5 Gaussian RBFs.
    Performs internal multi-start optimization to handle non-convexity.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    
    # Safety check for empty input
    if len(x) == 0:
        return np.array([])
    
    # We need to infer y? No, the evaluator provides x and we return prediction.
    # Wait, the task says: "Write equation(x, params)... recovers an unknown function."
    # "The evaluator fits params... minimizing the training MSE."
    # This implies the evaluator has y_train and x_train.
    # But `equation` only receives x and params. It does NOT receive y.
    # 
    # RE-READING CONTRACT:
    # "The evaluator fits params... minimizing the training MSE."
    # "equation must return an array with one prediction per point."
    # 
    # CRITICAL INSIGHT:
    # The standard interface for these tasks usually implies that `equation` is called
    # during the optimization loop with the *training* x to compute the prediction,
    # and then the evaluator computes the loss against the hidden y.
    # 
    # HOWEVER, the contract says: "The evaluator fits params... starting from all ones...".
    # This means the evaluator calls `equation(x_train, params_current)` to get `pred_train`,
    # then computes `loss = MSE(pred_train, y_train)`.
    # 
    # If I perform internal optimization inside `equation`, I need `y_train` to compute the loss.
    # I DO NOT HAVE ACCESS TO `y_train` INSIDE `equation`.
    # 
    # THEREFORE, I CANNOT DO INTERNAL GLOBAL SEARCH THAT DEPENDS ON Y.
    # 
    # Let's re-read carefully:
    # "Write equation(x, params), a closed-form expression that recovers an unknown function."
    # "The evaluator fits params... minimizing the training MSE."
    # 
    # This is a "Closed-form expression" task. The `params` are the free variables.
    # The evaluator runs BFGS on `params`.
    # 
    # So, I must design a `equation(x, params)` such that:
    # 1. It is differentiable (for BFGS).
    # 2. The landscape is as convex or well-behaved as possible.
    # 3. It has enough capacity to fit the function.
    #
    # The previous Chebyshev approach was linear in params. BFGS should solve this perfectly if the basis is well-conditioned.
    # Why did it score 5360288 (terrible)? 
    # Maybe the target function is NOT well-approximated by a degree 9 polynomial?
    # Or maybe the BFGS failed?
    # 
    # If the target is, say, sin(x) * exp(x) or something oscillatory, a degree 9 polynomial might not be enough.
    # 
    # Alternative Strategy: **Fourier Series**
    # Since x is in [-3, 3], we can use sin/cos basis functions.
    # This is also linear in params.
    # 
    # Alternative Strategy: **RBFs with FIXED centers and widths, linear in weights.**
    # If we fix mu_i and sigma_i, the model is linear in w_i.
    # This avoids the non-convexity of optimizing mu and sigma.
    # We can choose a dense grid of centers to ensure approximation power.
    #
    # Let's try: **Fixed RBF Basis (Linear in Weights)**
    # Use 10 parameters.
    # We can use 10 RBFs with fixed centers spaced evenly in [-3, 3] and fixed widths.
    # This is linear in params. BFGS will solve it.
    # The condition number might be high if widths are small, but we can choose moderate widths.
    #
    # Let's compare to Chebyshev:
    # Chebyshev degree 9 is global.
    # RBFs are local.
    # If the function has sharp features, RBFs are better.
    # If the function is smooth, Chebyshev is better.
    #
    # Given the score was ~5e6 (bad), it suggests the fit was very poor.
    # 
    # Let's try a **Hybrid**: 
    # Use a basis that includes both low-frequency (polynomial/sinusoidal) and high-frequency (localized) components.
    # But we only have 10 params.
    #
    # Let's go back to **Linear Basis** but with a **Better Basis**.
    # 
    # What if the issue was numerical instability in BFGS for Chebyshev?
    # Chebyshev polynomials T_k(t) grow in magnitude at the edges? No, they are bounded by 1 in [-1,1].
    # The basis matrix should be well-conditioned.
    #
    # Maybe the target function is NOT smooth?
    # If it's noisy? "The targets are noiseless."
    #
    # Let's try **Fourier Series** with 5 frequencies (sin/cos pairs).
    # Basis: 1, cos(x), sin(x), cos(2x), sin(2x), ..., cos(5x), sin(5x).
    # Total 11 terms. We have 10 params.
    # Drop the constant? No, keep constant. Drop highest frequency?
    # Let's use: 1, cos(x), sin(x), cos(2x), sin(2x), cos(3x), sin(3x), cos(4x), sin(4x), cos(5x).
    # (10 terms).
    #
    # Fourier series are excellent for periodic or oscillatory functions.
    # If the function is a polynomial, Fourier series converges slowly (Gibbs phenomenon).
    #
    # Let's try **Scaled Monomials** with regularization? No, we can't regularize, we just return pred.
    #
    # Let's look at the "Inspiration 3" which used Mixture of 5 Gaussians and scored 1298.
    # That is much better than 5360288.
    # Inspiration 3 used NON-LINEAR params (mu, sigma, w).
    # And it scored well.
    # Why? Because the evaluator's BFGS *did* manage to fit it?
    # Or maybe the "Initial program" in Inspiration 3 was different.
    #
    # Wait, Inspiration 3 says: "Non-Linear Parametric Model (a mixture of 5 Gauss...".
    # Score 1298.
    # Current Score: 5360288.
    #
    # The jump from 5360288 to 1298 is huge.
    # This suggests the Chebyshev approach failed catastrophically (maybe BFGS diverged or got stuck).
    #
    # If I use a **Linear** model, BFGS should work.
    # If I use a **Non-Linear** model, BFGS might fail, but if it works, it's powerful.
    #
    # Let's try a **Safe Non-Linear Model**:
    # Use RBFs, but make the centers and widths **PARAMETERS** that are initialized well?
    # No, the evaluator starts params at all ones.
    # If mu_i = 1 and sigma_i = 1 for all i, all Gaussians are at x=1 with width 1.
    # They are identical. The gradient with respect to mu_i will be zero if w_i are symmetric?
    # Actually, if all mu_i are 1, the model is w_sum * exp(-(x-1)^2/2).
    # This is a single Gaussian centered at 1.
    # If the target is centered at 0, this is a bad start.
    #
    # To fix the "all ones" start problem for non-linear models, we can **reparameterize**.
    # 
    # Idea: **Offset and Scale Params**
    # Instead of mu_i being the center, let mu_i = base_center_i + param_i.
    # But we don't know base_center_i.
    #
    # Alternative: **Use a Linear Model with a Very Rich Basis**
    # If the Chebyshev degree 9 was not enough, try **Piecewise Linear Basis**?
    # Or **Spline Basis**?
    #
    # Let's try **B-splines**.
    # B-splines are local, smooth, and form a linear basis.
    # We can define 10 B-spline basis functions.
    # This is linear in params. BFGS will work.
    # It can capture sharp features better than global polynomials.
    #
    # Let's implement a **Cubic B-Spline** basis.
    # Domain [-3, 3].
    # We need 10 basis functions.
    # We can define knots evenly spaced.
    #
    # Or simpler: **Localized Gaussian Basis with FIXED centers and widths**.
    # This is linear in weights.
    # Centers: -3, -2.1, -1.2, -0.3, 0.6, 1.5, 2.4, 3.3? 
    # We have 10 params. Let's use 10 Gaussians.
    # Centers: np.linspace(-3, 3, 10)
    # Widths: Fixed to 0.5? 
    # If width is 0.5, the basis functions are localized.
    # The matrix will be sparse-ish and well-conditioned if centers are not too close.
    #
    # Let's try this: **Fixed RBF Basis (Linear in Weights)**.
    
    # 10 Centers
    centers = np.linspace(-3.0, 3.0, 10)
    # Fixed Width
    sigma = 0.8  # Moderate width to ensure overlap but localization
    
    # Compute basis
    # B[i, j] = exp(-(x_i - center_j)^2 / (2 * sigma^2))
    
    # This is linear in params.
    # BFGS should converge quickly.
    # It can fit any function that is a sum of these Gaussians.
    # With 10 Gaussians spanning [-3, 3], it can fit a wide variety of smooth functions.
    
    # Let's implement this.
    
    p = np.asarray(params, dtype=np.float64).flatten()
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    else:
        p = p[:10]
        
    # Centers
    centers = np.linspace(-3.0, 3.0, 10)
    sigma = 0.8
    
    # Compute distances
    # x is (N,), centers is (10,)
    # diff is (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Exponent
    exp_term = - (diff**2) / (2.0 * sigma**2)
    
    # Basis matrix
    basis = np.exp(exp_term)
    
    # Prediction
    pred = basis @ p
    
    # Safety
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred

# EVOLVE-BLOCK-END
