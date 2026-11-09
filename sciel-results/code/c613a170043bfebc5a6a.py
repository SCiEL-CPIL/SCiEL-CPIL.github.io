"""
Strategy: Black-Box Non-Linear Search.

The current approach relies on BFGS to optimize parameters for a fixed or 
selected linear basis. This approach rethinks the problem: 
The `equation` function is called by the evaluator with `x` and `params`. 
While the evaluator uses BFGS to update `params`, we can ignore the 
`params` provided by BFGS (or use them only as a weak prior) and instead 
perform our own internal search for the best function form and its 
parameters directly inside `equation(x, params)`.

We maintain a library of candidate function templates. 
For each call, we quickly evaluate which template best fits the local 
structure of `x` (using a fast, grid-based or analytic search where 
possible) and return the predictions from that best template.

Since the function is called repeatedly during optimization with the 
same `x` (the training points), we can cache the best template and 
parameters found for that specific `x` configuration to ensure speed 
and consistency.

This allows us to jump out of local minima of the BFGS optimizer by 
effectively replacing the objective function landscape with a 
piecewise-constant approximation of the global optimum over a rich 
hypothesis space.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
from functools import lru_cache
import hashlib

# Deadline guard
_START_TIME = None

def _get_elapsed_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

# Cache for internal search results
# Key: hash of the input x array (rounded for stability)
# Value: (best_func_id, best_params_array)
_search_cache = {}

def _hash_x(x):
    """Create a stable hash for the input array."""
    # Round to 6 decimal places to handle floating point noise in x if any
    x_rounded = np.round(x, 6)
    # Use a simple hash of the bytes
    return hashlib.md5(x_rounded.tobytes()).hexdigest()

def _generate_candidates(x):
    """
    Generate a diverse set of candidate functions and evaluate their potential.
    Since we don't see y, we can't actually fit them here without y.
    WAIT: The equation function receives x and params, but NOT y.
    Therefore, we CANNOT fit the function to y inside equation().
    
    RE-EVALUATION OF STRATEGY:
    The evaluator calls equation(x, params) to get predictions.
    It computes MSE = mean((pred - y)^2).
    It uses BFGS to minimize this MSE by adjusting params.
    
    If we ignore `params` and return a fixed prediction based only on x, 
    BFGS will see a constant gradient (or zero gradient) and stop immediately.
    The score will be the MSE of that fixed prediction.
    
    So, we MUST use `params` to allow BFGS to adjust the output.
    
    New Strategy: 
    Use a "Switching" or "Gating" mechanism.
    We define a model: y = sum( w_i * f_i(x) * params_i ) ? No, params are fixed length 10.
    
    Let's look at the "Non-linear parametric model" inspiration.
    If we define equation(x, params) as a non-linear function of x and params, 
    BFGS can optimize params to fit the non-linear shape.
    
    The problem with the current program is that it selects a LINEAR basis.
    If the true function is non-linear (e.g., x^4 or sin(x)), a linear basis of 
    polynomials up to order 9 CAN fit it, but BFGS might struggle if the 
    condition number is bad or if the initial guess (all ones) is far from the 
    solution in the high-dimensional parameter space.
    
    However, the score 101239161194 is HUGE. This suggests the fit is terrible.
    Why?
    1. Maybe the basis selection is picking a bad basis (e.g., RBF with fixed width 
       that doesn't match the function).
    2. Maybe BFGS is failing to converge from the "all ones" start for these bases.
    3. Maybe the normalization is wrong.
    
    Let's try a DIFFERENT design family: 
    **Universal Approximator with Non-Linear Parameter Dependence.**
    
    Instead of linear basis @ params, let's use a model where params directly 
    control the shape of a flexible function, and the function is non-linear 
    in params. This can sometimes help BFGS find the solution faster if the 
    parameter space is better conditioned.
    
    Or, better yet: **Kernel Regression / RBF with Optimized Widths.**
    If we use RBFs, the centers and widths are fixed. Only amplitudes are 
    optimized. This is linear in params.
    
    What if we use a model like:
    y = params[0] * x + params[1] * x^2 + ... + params[9] * x^9
    This is standard polynomial regression. It should work for smooth functions 
    on [-2, 2]. Why is the score so bad?
    
    Perhaps the issue is that `np.linalg.cond` is expensive or unstable, or the 
    selection logic is flawed.
    
    Let's try a **Fixed, Robust Basis** that is known to be well-conditioned 
    and capable of representing a wide range of functions. 
    Chebyshev polynomials are the gold standard for polynomial approximation 
    on [-1, 1].
    
    The current program switches between bases. This introduces discontinuities 
    in the optimization landscape. If the basis switches between calls, BFGS 
    gets confused because the gradient changes direction abruptly.
    
    **CRITICAL INSIGHT:** 
    The evaluator calls `equation` many times during BFGS. If `equation` 
    changes the basis (the mapping from params to predictions) between calls, 
    BFGS will fail because it assumes a consistent function landscape.
    The current program calculates condition number EVERY call. If the 
    condition number fluctuates due to numerical noise or if the logic picks 
    different bases for slightly different internal states, BFGS breaks.
    
    **New Strategy:** 
    Use a SINGLE, FIXED, HIGH-QUALITY BASIS. 
    Specifically, use **Chebyshev Polynomials of the First Kind** on the 
    normalized domain [-1, 1]. 
    Chebyshev polynomials are orthogonal and have minimal Runge phenomenon.
    They are the best linear basis for approximating smooth functions on 
    [-1, 1].
    
    To handle non-smooth or non-polynomial functions, we can mix in some 
    trigonometric terms, but we must keep the basis FIXED.
    
    Let's define a fixed basis of 10 functions:
    1. T_0(x) = 1
    2. T_1(x) = x
    3. T_2(x) = 2x^2 - 1
    4. T_3(x) = 4x^3 - 3x
    5. T_4(x) = 8x^4 - 8x^2 + 1
    6. T_5(x) = 16x^5 - 20x^3 + 5x
    7. sin(x)  -- adds non-polynomial flexibility
    8. cos(x)  -- adds non-polynomial flexibility
    9. x^4     -- redundant with Chebyshev, but maybe helps conditioning? No, let's stick to orthogonal.
    10. x^5
    
    Actually, a mix of Chebyshev and Trig is good.
    Let's use:
    b[0] = 1
    b[1] = x_norm
    b[2] = x_norm^2
    b[3] = x_norm^3
    b[4] = x_norm^4
    b[5] = x_norm^5
    b[6] = sin(x)
    b[7] = cos(x)
    b[8] = sin(2x)
    b[9] = cos(2x)
    
    This is a FIXED basis. No switching.
    This ensures BFGS sees a consistent quadratic (or close to it) error 
    surface if the function is in the span, or a consistent non-convex 
    surface that is at least stable.
    
    To improve BFGS performance, we can scale the parameters or the basis.
    Chebyshev polynomials are bounded by 1 on [-1, 1]. 
    Monomials can grow. 
    Let's use Chebyshev polynomials for the first 6, and normalized trig for the rest.
    
    Also, the initial guess is "all ones". 
    If the true function is y = x, and our basis is [1, x, ...], 
    params = [1, 1, 1, ...] gives y_pred = 1 + x + ... which is far from x.
    BFGS needs to move params[0] to 0 and keep params[1] at 1.
    This is a large move.
    
    Can we help BFGS? 
    The task says: "The evaluator fits params... starting from all ones".
    We cannot change the start.
    
    However, we can make the basis functions such that "all ones" is a 
    reasonable start for a wider class of functions?
    If we use a basis where the average value is 0 (except the constant), 
    "all ones" might be better?
    
    Let's stick to the FIXED BASIS strategy but optimize the choice of basis 
    functions for maximum coverage and stability.
    
    Basis:
    1. 1
    2. x
    3. x^2
    4. x^3
    5. x^4
    6. x^5
    7. sin(x)
    8. cos(x)
    9. sin(2x)
    10. cos(2x)
    
    This covers polynomials up to degree 5 and low-frequency oscillations.
    It is fixed.
    It is smooth.
    It is bounded.
    
    Let's implement this.
    """
    pass

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fixed Basis Strategy.
    Uses a consistent set of 10 basis functions to ensure BFGS stability.
    """
    # Time check
    if _get_elapsed_time() > 140.0:
        # Fallback: simple linear
        return params[0] * x + params[1]

    x_flat = x.flatten()
    n_points = len(x_flat)
    
    if n_points == 0:
        return np.array([])

    try:
        # Normalize x to [-1, 1] for polynomial stability
        z = x_flat / 2.0
        
        # Construct the fixed basis matrix
        # We want a basis that is well-conditioned and covers a wide range of functions.
        # Chebyshev polynomials are ideal for the polynomial part.
        
        # T0 = 1
        # T1 = z
        # T2 = 2z^2 - 1
        # T3 = 4z^3 - 3z
        # T4 = 8z^4 - 8z^2 + 1
        # T5 = 16z^5 - 20z^3 + 5z
        
        b = np.zeros((n_points, 10))
        
        # Polynomial part (Chebyshev)
        T0 = np.ones_like(z)
        T1 = z
        T2 = 2 * z**2 - 1
        T3 = 4 * z**3 - 3 * z
        T4 = 8 * z**4 - 8 * z**2 + 1
        T5 = 16 * z**5 - 20 * z**3 + 5 * z
        
        b[:, 0] = T0
        b[:, 1] = T1
        b[:, 2] = T2
        b[:, 3] = T3
        b[:, 4] = T4
        b[:, 5] = T5
        
        # Trigonometric part (to capture non-polynomial smooth functions)
        # Use original x for trig to preserve frequency
        b[:, 6] = np.sin(x_flat)
        b[:, 7] = np.cos(x_flat)
        b[:, 8] = np.sin(2 * x_flat)
        b[:, 9] = np.cos(2 * x_flat)
        
        # Prediction
        pred = b @ params
        
        # Ensure finite
        if not np.all(np.isfinite(pred)):
             return np.zeros_like(x_flat)
             
        return pred

    except Exception:
        # Fallback
        return params[0] * x_flat + params[1]

# EVOLVE-BLOCK-END
