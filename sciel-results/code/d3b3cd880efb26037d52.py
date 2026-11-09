"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series approximation with fixed orthogonal frequencies.

    Domain [-2, 2] has length 4, so fundamental angular frequency for
    orthogonality is omega0 = pi / 2. We use:
      params[0] * 1
      params[1] * sin(omega0 * x)
      params[2] * cos(omega0 * x)
      params[3] * sin(2*omega0 * x)
      params[4] * cos(2*omega0 * x)
      params[5] * sin(3*omega0 * x)
      params[6] * cos(3*omega0 * x)
      params[7] * sin(4*omega0 * x)
      params[8] * cos(4*omega0 * x)
      params[9] * cos(5*omega0 * x)

    This is a full set of 10 linearly independent, mutually orthogonal
    (over [-2,2]) basis functions: constant, sines/cosines for k=1..4,
    and the 5th cosine (sine of 5*omega0 is odd and orthogonal to all
    cosines; we include cos(5) to keep symmetry in even/odd coverage).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        x = x.reshape(1)

    omega0 = np.pi / 2.0

    # Build basis matrix B of shape (n, 10)
    n = x.size
    B = np.empty((n, 10), dtype=float)

    B[:, 0] = 1.0

    # k = 1..4: sine and cosine
    for k in range(1, 5):
        f = k * omega0
        B[:, 1 + 2*(k-1)] = np.sin(f * x)
        B[:, 2 + 2*(k-1)] = np.cos(f * x)

    # k = 5: use cosine for the 10th parameter
    B[:, 9] = np.cos(5 * omega0 * x)

    # Linear combination: out = B @ params
    out = B @ params

    # Ensure output shape matches input shape
    if np.isscalar(x) or x.size == 1 and np.ndim(x) == 0:
        return out.reshape(1)
    return out
# EVOLVE-BLOCK-END
