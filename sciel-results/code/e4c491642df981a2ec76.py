"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the function using a 3rd order polynomial basis.
    Terms: 1, x, y, z, x^2, y^2, z^2, xy, xz, yz
    (Note: we only have 10 params, so we stick to 2nd order but try different combinations)
    Let's try: 1, x, y, z, x^2, y^2, z^2, x*y, x*z, y*z
    Wait, the previous one was 2nd order and got 7.39.
    The parent (linear) got 1.0.
    This suggests the function is NOT a simple polynomial or the 2nd order terms are hurting.
    If linear is 1.0 (perfect), why did 2nd order drop to 7.39?
    Maybe the fit is unstable or the 2nd order terms are causing overfitting or numerical issues?
    Or maybe the function is actually linear?
    If the function is linear, a 2nd order model should also fit it perfectly (coeffs of x^2 etc become 0).
    Unless... BFGS is getting stuck in a local minimum or the Hessian is singular/bad.
    Or maybe the "1.0" score from the parent was actually just the baseline and the 2nd order is overfitting to noise? But it says "noiseless".
    
    Let's look at the score definition: mean((prediction - y)^2) / var(y).
    If score is 1.0, it means the model explains 0% of the variance? No, usually 1.0 means predicting the mean.
    If the parent (linear) got 1.0, it failed to fit?
    Or is 1.0 a "perfect" score in this specific context? No, "lower is better".
    If score is 1.0, it's bad.
    If score is 0.0, it's perfect.
    
    So the linear model got 1.0 (bad).
    The 2nd order model got 7.39 (worse).
    
    This implies the function is NOT a low-degree polynomial.
    It could be exponential, trigonometric, or rational.
    
    Let's try a basis that includes exponentials or trig functions.
    Common functions: exp(x), sin(x), etc.
    
    Let's try:
    1, x, y, z, exp(x), exp(y), exp(z), x*y, x*z, y*z
    
    Or maybe:
    1, x, y, z, 1/x, 1/y, 1/z, x*y, x*z, y*z
    
    Let's try a mix of polynomial and exponential.
    
    Candidate 1: 1, x, y, z, exp(x), exp(y), exp(z), x^2, y^2, z^2
    Candidate 2: 1, x, y, z, sin(x), sin(y), sin(z), cos(x), cos(y), cos(z)
    Candidate 3: 1, x, y, z, x*y, x*z, y*z, x*y*z, x^2, y^2
    
    Given the inputs are in [0.5, 2], exp(x) ranges from ~1.6 to ~7.4.
    sin(x) ranges from ~0.4 to ~1.0.
    
    Let's try a basis with exponentials, as they are common in these "hidden function" tasks.
    
    Basis: 1, x, y, z, exp(x), exp(y), exp(z), x*y, x*z, y*z
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Compute basis elements
    ex = np.exp(x)
    ey = np.exp(y)
    ez = np.exp(z)
    xy = x * y
    xz = x * z
    yz = y * z
    
    # Linear combination
    return c0 + c1 * x + c2 * y + c3 * z + c4 * ex + c5 * ey + c6 * ez + c7 * xy + c8 * xz + c9 * yz
# EVOLVE-BLOCK-END
