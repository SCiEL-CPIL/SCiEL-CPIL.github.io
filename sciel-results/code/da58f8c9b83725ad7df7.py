"""Rewritten program for symbolic regression with 3 inputs."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Non-linear-in-params model: sum of exponential, polynomial, and reciprocal terms.
    
    Structure:
      pred = p0 * exp(p1*x) * exp(p2*y) * exp(p3*z)
           + p4 * x*y*z
           + p5 * x*y
           + p6 * x*z
           + p7 * y*z
           + p8 * (1/x + 1/y + 1/z)
           + p9
    
    The exponential part is non-linear in p1,p2,p3, making the overall
    optimization non-convex. This allows capturing exponential targets
    that a purely linear basis cannot represent.
    
    Inputs are in [0.5, 2], so exp(p*x) is bounded and finite for
    reasonable param values. We clip exponent arguments for safety.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Clip exponent arguments to prevent overflow
    # x,y,z in [0.5, 2], so p*x in [0.5*p, 2*p]
    # For p in [-10, 10], p*x in [-20, 20], exp(20) ~ 4.8e8 which is fine
    # But to be safe, clip to [-50, 50]
    ex = np.clip(p1 * x, -50.0, 50.0)
    ey = np.clip(p2 * y, -50.0, 50.0)
    ez = np.clip(p3 * z, -50.0, 50.0)
    
    # Exponential term (non-linear in p1, p2, p3)
    exp_term = p0 * np.exp(ex) * np.exp(ey) * np.exp(ez)
    
    # Polynomial interaction terms (linear in p4-p7)
    poly_term = (p4 * x * y * z +
                 p5 * x * y +
                 p6 * x * z +
                 p7 * y * z)
    
    # Reciprocal term (linear in p8)
    recip_term = p8 * (1.0 / x + 1.0 / y + 1.0 / z)
    
    # Constant
    const = p9
    
    pred = exp_term + poly_term + recip_term + const
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), 0.0, pred)
    
    return pred
# EVOLVE-BLOCK-END
