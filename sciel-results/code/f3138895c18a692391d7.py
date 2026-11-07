"""Symbolic regression: rational basis with tanh-bounded denominator for BFGS stability."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational basis model with 10 parameters, reparameterized for BFGS stability.
    
    f(x, y) = (p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*xy) / (1 + a6*x + a7*y + a8*x^2 + a9*y^2)
    
    where a_i = 0.5 * tanh(p_i) for i in 6..9, bounding denominator coefficients to [-0.5, 0.5].
    
    Key property: With |a_i| <= 0.5 and x, y in [-1, 1]:
      den = 1 + a6*x + a7*y + a8*x^2 + a9*y^2
      Minimum possible value: 1 - 0.5*1 - 0.5*1 + 0.5*0 + 0.5*0 = 0 (when a6=a7=0.5, x=y=-1, a8=a9=0)
      Wait: 1 + 0.5*(-1) + 0.5*(-1) + 0*1 + 0*1 = 1 - 0.5 - 0.5 = 0.
      So den can approach 0. We need a safety margin.
      
      Actually with a8, a9 terms: at x=y=-1, x^2=y^2=1:
      den = 1 + a6*(-1) + a7*(-1) + a8*(1) + a9*(1)
      Min when a6=a7=0.5, a8=a9=-0.5: 1 - 0.5 - 0.5 - 0.5 - 0.5 = -1.0
      That's negative! So we need to be more careful.
      
      Let me use smaller bounds: a_i = 0.3 * tanh(p_i), so |a_i| <= 0.3.
      Min den: 1 - 0.3 - 0.3 - 0.3 - 0.3 = -0.2. Still can be negative.
      
      Better: use a_i = 0.2 * tanh(p_i), |a_i| <= 0.2.
      Min den: 1 - 0.2 - 0.2 - 0.2 - 0.2 = 0.2. Always positive!
      
      So with 0.2 * tanh, den >= 0.2 always. No clipping needed, smooth everywhere.
    """
    p = params

    # Pre-compute powers
    x2 = x * x
    y2 = y * y
    xy = x * y

    # Numerator: p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*xy
    num = p[0] + p[1] * x + p[2] * y + p[3] * x2 + p[4] * y2 + p[5] * xy

    # Denominator with bounded coefficients via tanh
    # a_i = 0.2 * tanh(p_i) ensures |a_i| <= 0.2
    # This guarantees den >= 1 - 0.2*4 = 0.2 > 0 for all x,y in [-1,1]
    a6 = 0.2 * np.tanh(p[6])
    a7 = 0.2 * np.tanh(p[7])
    a8 = 0.2 * np.tanh(p[8])
    a9 = 0.2 * np.tanh(p[9])
    
    den = 1.0 + a6 * x + a7 * y + a8 * x2 + a9 * y2

    # den is guaranteed >= 0.2, so no clipping needed. Smooth everywhere.
    pred = num / den
    
    # Ensure finite values (safety net for extreme parameter values)
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
