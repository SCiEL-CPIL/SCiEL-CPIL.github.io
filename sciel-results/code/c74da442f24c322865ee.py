"""Symbolic regression with a rich basis: polynomial + trig + sigmoid + Gaussian."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model combining:
      - Low-order polynomial (up to x^3)
      - Trigonometric terms (sin(x), cos(x))
      - A learnable sigmoid (center, steepness, amplitude)
      - A Gaussian bump (center, width, amplitude)
    
    params (length 10):
      p0: constant
      p1: linear coeff
      p2: quadratic coeff
      p3: cubic coeff
      p4: sin(x) coeff
      p5: cos(x) coeff
      p6: sigmoid center
      p7: sigmoid steepness (positive magnitude)
      p8: sigmoid amplitude
      p9: Gaussian width (positive magnitude)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Polynomial part
    poly = p0 + p1 * x + p2 * x**2 + p3 * x**3

    # Trigonometric part
    trig = p4 * np.sin(x) + p5 * np.cos(x)

    # Sigmoid part: amplitude * sigmoid(steepness * (x - center))
    # Use a safe sigmoid to avoid overflow
    z = p7 * (x - p6)
    # Clip z to avoid extreme values in exp
    z = np.clip(z, -50, 50)
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    sig_part = p8 * sigmoid

    # Gaussian bump: amplitude is implicit via p8 reuse? No, we need another param.
    # We only have 10 params. Let's reassign:
    # Actually, let's drop the Gaussian and keep the sigmoid, or combine differently.
    # Let's use the 10 params as:
    # p0: const, p1: lin, p2: quad, p3: cubic
    # p4: sin, p5: cos
    # p6: sigmoid center, p7: sigmoid steep, p8: sigmoid amp
    # p9: another sigmoid center? Or a second Gaussian?
    # Let's keep it simple: just the sigmoid with 3 params and a linear/quadratic/trig base.
    # That uses 6 + 3 = 9 params. We have 10. Let's add a second trig term or a higher poly.
    # Let's add x^4 term as p9.
    
    # Recompute with p9 as x^4 coeff
    poly4 = p9 * x**4

    return poly + trig + sig_part + poly4
# EVOLVE-BLOCK-END
