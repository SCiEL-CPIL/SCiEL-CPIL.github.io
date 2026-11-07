# EVOLVE-BLOCK-START
import time
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form candidate: fixed smooth basis (Fourier + polynomial + radial),
    10 learnable linear coefficients fitted by the external BFGS.

    Basis (10 columns):
      0: 1
      1: x
      2: y
      3: x*y
      4: sin(x + y)
      5: cos(x + y)
      6: sin(3x + 2y)
      7: cos(3x + 2y)
      8: exp(-2*(x^2 + y^2))
      9: exp(-5*(x^2 + y^2))

    Linear-in-params => smooth objective => BFGS-friendly.
    """
    # deadline guard (safety margin well under the 150s kill)
    t0 = time.time()

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size

    # ---- build fixed basis ----
    B = np.empty((n, 10), dtype=float)
    B[:, 0] = 1.0
    B[:, 1] = x
    B[:, 2] = y
    B[:, 3] = x * y

    s1 = x + y
    B[:, 4] = np.sin(s1)
    B[:, 5] = np.cos(s1)

    s2 = 3.0 * x + 2.0 * y
    B[:, 6] = np.sin(s2)
    B[:, 7] = np.cos(s2)

    r2 = x * x + y * y
    B[:, 8] = np.exp(-2.0 * r2)
    B[:, 9] = np.exp(-5.0 * r2)

    # ---- linear combination ----
    p = np.asarray(params, dtype=float).ravel()
    if p.size < 10:
        # pad with zeros defensively
        p = np.pad(p, (0, 10 - p.size))
    out = B @ p[:10]

    # safety: replace any non-finite with 0 (should not happen)
    if not np.all(np.isfinite(out)):
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    # deadline guard: if we are dangerously close to the limit, return now
    if time.time() - t0 > 140.0:
        return out

    return out
# EVOLVE-BLOCK-END
