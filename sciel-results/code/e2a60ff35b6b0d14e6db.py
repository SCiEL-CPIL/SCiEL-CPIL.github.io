import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Feature-basis oscillator model with normalized inputs.
    
    Uses a weighted sum of bounded, normalized polynomial features to
    provide BFGS with a well-conditioned optimization landscape.
    
    Features (all bounded via tanh or trig to prevent runaway growth):
      f0 = 1                          (offset)
      f1 = tanh(x)                    (linear position, bounded)
      f2 = tanh(v)                    (linear velocity, bounded)
      f3 = cos(t)                     (driving, in-phase)
      f4 = sin(t)                     (driving, quadrature)
      f5 = tanh(x) * tanh(v)          (position-velocity coupling)
      f6 = tanh(x) * tanh(x)          (quadratic position)
      f7 = tanh(v) * tanh(v)          (quadratic velocity)
      f8 = tanh(x) * cos(t)           (position-driven coupling)
      f9 = tanh(v) * sin(t)           (velocity-driven coupling)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    tx = np.tanh(x)
    tv = np.tanh(v)
    ct = np.cos(t)
    st = np.sin(t)
    
    return (p0
            + p1 * tx
            + p2 * tv
            + p3 * ct
            + p4 * st
            + p5 * tx * tv
            + p6 * tx * tx
            + p7 * tv * tv
            + p8 * tx * ct
            + p9 * tv * st)
