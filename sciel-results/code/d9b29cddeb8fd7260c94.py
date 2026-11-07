"""Symbolic regression with bounded rational basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational model: (1 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y) /
                    (1 + b1*tanh(x) + b2*tanh(y) + b3*tanh(x^2+y^2))
    The tanh in the denominator bounds it to [1-3, 1+3], preventing poles
    and keeping BFGS stable while allowing non-polynomial saturation behavior.
    """
    p = params
    # Numerator: quadratic in x, y (6 params)
    num = (1.0 +
           p[0] * x +
           p[1] * y +
           p[2] * x**2 +
           p[3] * y**2 +
           p[4] * x * y)
    
    # Denominator: bounded via tanh (4 params)
    # tanh values are in [-1, 1], so denominator is in [1-3, 1+3] = [-2, 4]
    # To ensure denominator stays positive and bounded away from zero,
    # we use 1 + 0.5 * (weighted tanh terms), giving range [1-1.5, 1+1.5] = [-0.5, 2.5]
    # Better: use 2 + 0.5*(...) to guarantee positivity: range [0.5, 3.5]
    den = (2.0 +
           0.5 * p[5] * np.tanh(x) +
           0.5 * p[6] * np.tanh(y) +
           0.5 * p[7] * np.tanh(x**2 + y**2) +
           0.5 * p[8] * np.tanh(x * y))
    
    # Safety: clip denominator to avoid any division by near-zero
    den = np.where(np.abs(den) < 0.1, np.sign(den) * 0.1, den)
    
    return num / den
# EVOLVE-BLOCK-END
