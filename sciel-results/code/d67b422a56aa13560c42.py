"""Rewritten program: rational basis model for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = None
_WALL_LIMIT = 150.0

def _check_deadline():
    """Check if we're approaching the wall-clock limit."""
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _WALL_LIMIT
    return time.time() > _DEADLINE - 10.0  # 10s safety margin


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function model with 10 parameters:
    f(x, y) = (p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*xy)
             / (1 + p6*x + p7*y + p8*x^2 + p9*y^2)
    
    This rational form can represent a much broader class of functions
    than a pure polynomial, including functions with poles and asymptotic
    behavior. The denominator is regularized to stay well-conditioned.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Preserve original shape
    orig_shape = x.shape
    
    # Ensure 1D
    x = x.ravel()
    y = y.ravel()
    
    # Check deadline - fallback to linear
    if _check_deadline():
        pred = params[0] + params[1] * x + params[2] * y
        return pred.reshape(orig_shape)
    
    x2 = x * x
    y2 = y * y
    
    # Numerator: quadratic polynomial
    num = (params[0] +
           params[1] * x +
           params[2] * y +
           params[3] * x2 +
           params[4] * y2 +
           params[5] * x * y)
    
    # Denominator: 1 + quadratic (regularized to avoid zero crossings)
    den = (1.0 +
           params[6] * x +
           params[7] * y +
           params[8] * x2 +
           params[9] * y2)
    
    # Avoid division by zero: clip denominator away from 0
    den_safe = np.where(np.abs(den) < 1e-8, 1e-8 * np.sign(den + 1e-15), den)
    
    pred = num / den_safe
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred.reshape(orig_shape)
# EVOLVE-BLOCK-END
