"""Symbolic regression with a rich, well-conditioned basis and safe evaluation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y from x using a 10-term basis.
    Basis:
      0: 1
      1: x
      2: x^2
      3: x^3
      4: sin(x)
      5: cos(x)
      6: exp(-x^2)   (Gaussian bump, well-behaved on [-3,3])
      7: tanh(x)
      8: abs(x)
      9: x * sin(x)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.ravel()

    # Guard against non-finite inputs
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)

    p = np.asarray(params, dtype=float)
    if p.shape != (10,):
        p = np.resize(p, 10)
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)

    # Compute basis columns
    x2 = x * x
    basis0 = np.ones_like(x)
    basis1 = x
    basis2 = x2
    basis3 = x2 * x
    basis4 = np.sin(x)
    basis5 = np.cos(x)
    basis6 = np.exp(-x2)
    basis7 = np.tanh(x)
    basis8 = np.abs(x)
    basis9 = x * basis4

    result = (
        p[0] * basis0
        + p[1] * basis1
        + p[2] * basis2
        + p[3] * basis3
        + p[4] * basis4
        + p[5] * basis5
        + p[6] * basis6
        + p[7] * basis7
        + p[8] * basis8
        + p[9] * basis9
    )

    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result
# EVOLVE-BLOCK-END
