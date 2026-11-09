# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a rational (Padé-like) function.
    
    The model is y = N(x, v) / D(x, v).
    The numerator captures linear and cubic nonlinearities typical of oscillators.
    The denominator provides saturation/bounding behavior and improves conditioning
    by preventing unbounded growth of the prediction, which helps BFGS convergence.
    
    Inputs:
        x: position
        v: velocity
        params: array of 10 parameters
    
    Returns:
        acceleration array
    """
    p = params
    
    # Scale inputs to O(1) range for better conditioning of polynomial coefficients
    # x range ~ [-0.57, 0.53], v range ~ [-0.18, 0.28]
    xs = x * 2.0
    vs = v * 5.0
    
    # Precompute powers
    xs2 = xs * xs
    vs2 = vs * vs
    xs3 = xs2 * xs
    vs3 = vs2 * vs
    xs4 = xs2 * xs2
    vs4 = vs2 * vs2
    
    # Numerator: Linear + Cubic + Mixed terms
    # p[0]: linear x
    # p[1]: linear v
    # p[2]: cubic x
    # p[3]: cubic v
    # p[4]: mixed x*v
    N = (p[0] * xs +
         p[1] * vs +
         p[2] * xs3 +
         p[3] * vs3 +
         p[4] * xs * vs)
         
    # Denominator: 1 + Quadratic + Higher order terms
    # Ensures D is near 1 for small inputs, preventing division by zero issues
    # and providing a smooth saturation effect for larger inputs.
    D = (1.0 +
         p[5] * xs2 +
         p[6] * vs2 +
         p[7] * xs2 * vs2 +
         p[8] * xs4 +
         p[9] * vs4)
         
    # Avoid division by zero (though D should be close to 1)
    # Clip D to a small epsilon to prevent inf/nan if parameters go wild
    # np.where is fast and safe
    D_safe = np.where(np.abs(D) < 1e-8, 1e-8, D)
    
    y = N / D_safe
    
    return y
# EVOLVE-BLOCK-END
