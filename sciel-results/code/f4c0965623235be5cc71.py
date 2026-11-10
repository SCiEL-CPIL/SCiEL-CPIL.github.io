"""Symbolic regression with rational function model."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function model: y(x) = P(x) / Q(x)
    where P(x) = params[0]*x^4 + params[1]*x^3 + params[2]*x^2 + params[3]*x + params[4]
          Q(x) = params[5]*x^4 + params[6]*x^3 + params[7]*x^2 + params[8]*x + params[9]
    
    To avoid division by zero and numerical instability, we add a small regularization
    to the denominator. The BFGS optimizer will fit the 10 parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Build numerator polynomial
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    numerator = (p[0] * x4 + p[1] * x3 + p[2] * x2 + p[3] * x + p[4])
    
    # Build denominator polynomial
    denominator = (p[5] * x4 + p[6] * x3 + p[7] * x2 + p[8] * x + p[9])
    
    # Add small regularization to prevent division by zero
    # Use a smooth regularization that doesn't disturb the fit too much
    eps = 1e-8
    # We want to ensure denominator is never zero. 
    # A common trick: denominator + sign(denominator)*eps won't work smoothly.
    # Instead, use denominator^2 + eps in a safe way? No, that changes the function class.
    # Better: just clip the denominator away from zero.
    # But clipping is non-differentiable.
    # Alternative: add a small constant to denominator. This shifts the function slightly.
    # Let's use: result = numerator / (denominator + eps * sign(denominator))
    # But sign is non-differentiable at 0.
    # 
    # Actually, for BFGS, if the denominator crosses zero, the loss becomes infinite,
    # and the optimizer should avoid that region. But to be safe, let's use a smooth
    # approximation: denominator / sqrt(denominator^2 + eps^2) * (denominator + eps)
    # This is getting complicated. Let's just add a tiny constant.
    # If the true function has no poles in [-2,2], this should be fine.
    
    denom_safe = denominator + eps * np.sign(denominator)
    # Handle the case where denominator is exactly 0
    denom_safe = np.where(np.abs(denominator) < 1e-12, eps, denom_safe)
    
    result = numerator / denom_safe
    
    # Ensure output is finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
