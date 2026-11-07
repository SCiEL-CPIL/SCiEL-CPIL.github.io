import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Chebyshev-polynomial + trigonometric basis (10 terms).
    
    The Chebyshev terms (0-4) provide a well-conditioned polynomial basis
    on [-1,1] that BFGS can fit easily from the all-ones start.
    The trigonometric terms (5-9) capture any periodic/oscillatory structure
    in the hidden function that pure polynomials miss.
    
    Basis:
      0: 1                    (constant)
      1: T1(x) = x            (linear x)
      2: T1(y) = y            (linear y)
      3: T2(x) = 2x^2-1       (quadratic x)
      4: T2(y) = 2y^2-1       (quadratic y)
      5: sin(pi*x)            (trig x)
      6: sin(pi*y)            (trig y)
      7: cos(pi*x)            (trig x)
      8: cos(pi*y)            (trig y)
      9: xy                   (interaction)
    """
    # Chebyshev polynomials of the first kind
    T1_x = x
    T1_y = y
    T2_x = 2.0 * x**2 - 1.0
    T2_y = 2.0 * y**2 - 1.0

    # Trigonometric terms
    sin_px = np.sin(np.pi * x)
    sin_py = np.sin(np.pi * y)
    cos_px = np.cos(np.pi * x)
    cos_py = np.cos(np.pi * y)

    # Interaction term
    xy = x * y

    f = (params[0] * 1.0 +
         params[1] * T1_x +
         params[2] * T1_y +
         params[3] * T2_x +
         params[4] * T2_y +
         params[5] * sin_px +
         params[6] * sin_py +
         params[7] * cos_px +
         params[8] * cos_py +
         params[9] * xy)

    return f
