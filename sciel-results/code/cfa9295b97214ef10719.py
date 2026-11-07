"""Symbolic regression for a nonlinear oscillator using a rational function basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# Strategy:
# Instead of a linear combination of basis functions (which is convex and limited to polynomial growth),
# we use a rational function form:
# y = N(x, v) / D(x, v)
# where N and D are linear combinations of a small set of basis functions.
# We have 10 parameters total. We split them: 5 for the numerator, 5 for the denominator.
# Basis functions:
# 1. 1 (constant)
# 2. x
# 3. v
# 4. x^2
# 5. v^2
#
# N(x,v) = p0 + p1*x + p2*v + p3*x^2 + p4*v^2
# D(x,v) = p5 + p6*x + p7*v + p8*x^2 + p9*v^2
#
# This allows the model to represent functions like:
# - Linear: if p5=1, p6..p9=0, it's linear.
# - Reciprocal: if N is constant and D is linear.
# - Hyperbolic tangent approximations or other bounded nonlinearities via ratios.
# - Duffing-like behavior with saturation effects if D grows faster than N.
#
# This is a non-convex problem for BFGS, but often finds better fits for physical systems
# with saturation or asymptotic behavior than pure polynomials.

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict acceleration using a rational function basis.
    
    Args:
        x: Position array
        v: Velocity array
        params: Array of 10 fitted constants [n0, n1, n2, n3, n4, d0, d1, d2, d3, d4]
        
    Returns:
        Prediction array
    """
    # Ensure x and v are 1D arrays
    x = np.asarray(x).flatten()
    v = np.asarray(v).flatten()
    
    # Extract parameters
    # Numerator coefficients
    n0 = params[0]
    n1 = params[1]
    n2 = params[2]
    n3 = params[3]
    n4 = params[4]
    
    # Denominator coefficients
    d0 = params[5]
    d1 = params[6]
    d2 = params[7]
    d3 = params[8]
    d4 = params[9]
    
    # Pre-compute powers
    x2 = x * x
    v2 = v * v
    
    # Construct Numerator
    numerator = n0 + n1 * x + n2 * v + n3 * x2 + n4 * v2
    
    # Construct Denominator
    denominator = d0 + d1 * x + d2 * v + d3 * x2 + d4 * v2
    
    # Handle division by zero or near-zero to prevent numerical instability
    # Add a small epsilon to the denominator to avoid inf/nan
    # We use a smooth regularization: denom_safe = denominator + epsilon * sign(denominator)
    # Or simpler: np.where(abs(denominator) < eps, eps * sign, denominator)
    # Since BFGS might explore regions where D=0, we need robustness.
    # A common trick is to add a small constant to the denominator magnitude if it's too small.
    # However, adding a constant changes the function. 
    # Better: use np.divide with where clause or just add a tiny epsilon.
    # Given the training range is bounded and target is small, 
    # the denominator likely stays away from 0 if the fit is good.
    # But during optimization, it might cross 0.
    # Let's use a soft regularization: denom_safe = denominator + 1e-8 * np.sign(denominator)
    # Actually, standard np.where is safer for gradients if not careful, but BFGS uses finite differences or analytic gradients? 
    # The evaluator uses scipy.optimize.minimize(method="BFGS"). 
    # If the function is not smooth at 0, BFGS might struggle.
    # Let's try a simple epsilon addition first, it's smooth.
    
    eps = 1e-8
    # To preserve sign and avoid discontinuity in derivative, we can do:
    # denom_safe = denominator + eps * np.tanh(denominator / eps) 
    # This is smooth and approximates denominator + sign(denominator)*eps for large denom, and 0 for small? No.
    # Simpler: just add eps. It shifts the function slightly but ensures stability.
    # Given the scale of inputs ([-0.5, 0.5]) and target ([-0.1, 0.1]), 
    # the denominator terms (x, v, x^2, v^2) are small. 
    # If d0 is close to 0, we have issues. 
    # Let's initialize params to all ones in the evaluator, so d0=1 initially.
    
    denom_safe = denominator + eps
    
    pred = numerator / denom_safe
    
    return pred
# EVOLVE-BLOCK-END
