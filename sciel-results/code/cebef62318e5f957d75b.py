"""Symbolic regression with separable product basis."""
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Separable product basis: f(x,y,z) ≈ (a0 + a1*x + a2*x^2) * (b0 + b1*y) * (c0 + c1*z)
    
    This uses all 10 parameters:
      p[0], p[1], p[2] = coefficients for x factor: a0, a1, a2
      p[3], p[4]       = coefficients for y factor: b0, b1
      p[5], p[6]       = coefficients for z factor: c0, c1
      p[7], p[8], p[9] = additional scaling/offset terms for flexibility
    
    Actually, let's use a richer basis that still fits in 10 params.
    Let's try: f = (a0 + a1*x + a2*x^2) * (b0 + b1*y) * (c0 + c1*z)
    That's 3 + 2 + 2 = 7 params. We have 10, so we can add more.
    
    Let me use: f = (a0 + a1*x + a2*x^2) * (b0 + b1*y + b2*y^2) * (c0 + c1*z)
    That's 3 + 3 + 2 = 8 params. Still 2 left.
    
    Or: f = (a0 + a1*x + a2*x^2) * (b0 + b1*y) * (c0 + c1*z) + d0 + d1*x + d2*y
    That's 3 + 2 + 2 + 3 = 10 params.
    
    Let me go with a clean separable quadratic:
    f(x,y,z) = (a0 + a1*x + a2*x^2) * (b0 + b1*y) * (c0 + c1*z)
    with 7 params, and use the remaining 3 for a linear correction term.
    
    Actually, let me just use a full quadratic polynomial in 3 variables:
    f = w0 + w1*x + w2*y + w3*z + w4*x^2 + w5*y^2 + w6*z^2 + w7*x*y + w8*x*z + w9*y*z
    That's exactly 10 params and captures all quadratic interactions.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    p = params
    
    # Full quadratic polynomial with 10 terms
    # f = p0 + p1*x + p2*y + p3*z + p4*x^2 + p5*y^2 + p6*z^2 + p7*x*y + p8*x*z + p9*y*z
    
    result = (p[0] + p[1]*x + p[2]*y + p[3]*z + 
              p[4]*x*x + p[5]*y*y + p[6]*z*z + 
              p[7]*x*y + p[8]*x*z + p[9]*y*z)
    
    return result
