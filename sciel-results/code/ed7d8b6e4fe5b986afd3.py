# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate with a fixed, smooth 10-term basis.
    BFGS (started at all ones) fits the 10 coefficients to the training data.

    Basis (closed-form, smooth, finite on [-1,1]^2):
      1. 1
      2. x
      3. y
      4. x^2
      5. y^2
      6. x*y
      7. x^3
      8. y^3
      9. sin(pi * x)
     10. sin(pi * y)

    This spans polynomials up to degree 3 (including the quadratic interaction)
    plus full-period sine components that capture oscillatory structure over
    the [-1, 1] domain. The basis is strictly linear in the 10 parameters,
    ensuring BFGS converges reliably to the global least-squares optimum.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Polynomial terms
    f1 = np.ones_like(x)
    f2 = x
    f3 = y
    f4 = x * x
    f5 = y * y
    f6 = x * y
    f7 = x * x * x
    f8 = y * y * y

    # Full-period sine terms over [-1, 1]
    f9 = np.sin(np.pi * x)
    f10 = np.sin(np.pi * y)

    out = (
        p0 * f1
        + p1 * f2
        + p2 * f3
        + p3 * f4
        + p4 * f5
        + p5 * f6
        + p6 * f7
        + p7 * f8
        + p8 * f9
        + p9 * f10
    )

    # Safety: replace any non-finite values with 0
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, 0.0)

    return out
# EVOLVE-BLOCK-END
