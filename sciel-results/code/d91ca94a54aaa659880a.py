"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Global state to persist across calls if the evaluator reuses the function object
# and to enforce the time budget.
_start_time = None
_best_result_cache = None
_cache_key = None

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive RBF model with self-contained optimization.
    
    Strategy:
    1. Ignore the `params` input for the initial structure (centers/widths) but use 
       the time budget to perform a local search for the optimal RBF parameters.
    2. The model is: y = sum_{i=1}^K w_i * exp(-beta_i * (x - c_i)^2)
    3. We use a fixed number of centers K=5 to balance expressiveness and optimization time.
    4. We optimize the centers (c), widths (beta), and weights (w) simultaneously using 
       Nelder-Mead, which is robust to non-differentiable noise and does not require 
       gradients (unlike BFGS which the evaluator uses for its own fit, but we are 
       bypassing that by doing our own fit inside the function).
       
    Wait, the evaluator calls equation(x, params) and expects the return to be the prediction 
    for the given x, using the params provided by the evaluator's BFGS?
    
    RE-READING CONTRACT:
    "The evaluator fits params, an array of length 10, with scipy.optimize.minimize... 
     equation must return an array with one prediction per point."
     
    CRITICAL INSIGHT:
    The evaluator calls `equation(x, params)` where `params` are the 10 values it is optimizing.
    The evaluator does NOT pass `y`. It only passes `x` and `params`.
    Therefore, I CANNOT fit the model to `y` inside `equation` because I don't have access to `y` 
    during the test phase or the training phase evaluation of the function itself.
    
    The evaluator's BFGS loop works like this:
    1. It has training data (x_train, y_train).
    2. It calls minimize(objective, x0=np.ones(10)).
    3. Inside objective, it calls equation(x_train, params_candidate) to get predictions.
    4. It computes MSE(predictions, y_train).
    5. It returns this MSE to BFGS.
    6. BFGS updates params_candidate.
    
    So, `equation(x, params)` is called MANY times during the training fit, with DIFFERENT `params` 
    each time. I do NOT have access to `y` inside `equation`.
    
    Therefore, my previous idea of "fitting inside equation" is INVALID because I don't have y.
    
    I must define a function f(x, params) such that when the evaluator runs BFGS on params, 
    the resulting params allow f(x, params) to approximate the hidden target function well.
    
    The current program uses a linear basis: f(x, p) = sum p_i * phi_i(x).
    This is linear in params. BFGS will find the least squares solution (or close to it).
    
    Why is the score ~4300? 
    Score = normalized MSE. 1.0 is perfect (or seed level). 
    If the seed is 1.0, and we are at 4300, we are very bad.
    This suggests the target function is NOT well-approximated by the current basis, 
    OR the optimization is failing, OR the basis is completely wrong.
    
    Given the "symbolic regression" context, the target is likely a simple formula 
    like sin(x), x^2, x^3, exp(x), etc.
    
    If the target is, say, exp(x), a polynomial basis up to degree 6 might not be enough 
    on [-3, 3] to get a very low error, but 4300x variance is huge.
    
    Let's look at the inspirations.
    Inspiration 1: "self-contained multi-start local search inside the equation".
    How can you do local search inside equation without y?
    You can't fit to y. But you CAN change the structure of the function based on x 
    in a way that is non-linear in params, making the landscape for BFGS different?
    No, BFGS optimizes params. If the function is non-linear in params, BFGS might 
    get stuck in local minima.
    
    However, if I use a basis that is more flexible, maybe BFGS will find a better solution.
    
    Alternative: What if the target is not a function of x alone, but the "symbolic regression" 
    implies it is.
    
    Let's rethink the "linear in params" constraint.
    If I use a linear basis, BFGS is guaranteed to converge to the global optimum 
    (least squares) for that basis. The only way to improve is to have a better basis.
    
    The current basis: 1, x, x^2, x^3, sin(x), cos(x), tanh(x), x^4/10, x^5/100, x^6/1000.
    
    Maybe the target is something like x * sin(x) or exp(-x^2)?
    exp(-x^2) is not in the span of {1, x, ..., x^6, sin, cos, tanh}.
    It can be approximated, but poorly with low-order terms.
    
    A better basis for general functions might include Gaussian RBFs with FIXED centers 
    and widths, where only the weights are optimized.
    
    If I fix the centers and widths, the model is STILL LINEAR in the weights (params).
    f(x, p) = sum p_i * exp(-(x - c_i)^2 / (2 * sigma_i^2))
    
    If I choose a set of centers covering [-3, 3] and appropriate widths, this basis 
    is much more flexible for local features than polynomials.
    
    Let's try a basis of 10 Gaussian RBFs with fixed centers and widths.
    Centers: evenly spaced from -3 to 3.
    Widths: constant or varying.
    
    This is a structural change: from Polynomial/Trig basis to RBF basis.
    It remains linear in params, so BFGS will still converge to the optimal weights for 
    this new basis. If the target is, say, a sharp peak or a complex wave, RBFs with 
    appropriate centers will approximate it much better than a degree 6 polynomial.
    
    Let's define:
    Centers: np.linspace(-3, 3, 10)
    Widths: 0.5 (sigma). This gives a spread of ~1 unit, so they overlap slightly, 
    providing smooth interpolation.
    
    f(x, p) = sum_{i=0}^{9} p[i] * exp(-(x - centers[i])**2 / (2 * 0.5**2))
    
    This is linear in p. BFGS will find the best p.
    
    Why would this be better?
    Polynomials are global. If the target has local features, polynomials need high 
    degree to capture them, leading to oscillation (Runge's phenomenon) and ill-conditioning.
    RBFs are local. They can represent local variations without affecting distant regions 
    as much.
    
    Let's implement this.
    
    """
    global _start_time, _best_result_cache, _cache_key
    
    # No caching needed for linear basis as BFGS handles the optimization.
    # We just need to return the prediction based on the provided params.
    
    x = np.asarray(x, dtype=float)
    p = params
    
    # Define RBF centers and widths
    # 10 params -> 10 RBFs
    centers = np.array([-3.0, -2.333, -1.666, -1.0, -0.333, 0.333, 1.0, 1.666, 2.333, 3.0])
    # Use a width that allows overlap. Sigma = 0.6
    sigma = 0.6
    widths_sq = 2.0 * sigma**2
    
    # Compute RBF values
    # x is shape (N,), centers is shape (10,)
    # diff is shape (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_vals = np.exp(-(diff**2) / widths_sq)
    
    # Linear combination
    # rbf_vals shape (N, 10), p shape (10,)
    result = rbf_vals @ p
    
    return result
# EVOLVE-BLOCK-END
