"""Physics-informed anisotropic RBF model for a damped nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator.

    Uses 3 anisotropic Gaussian RBF terms in the (x, v) plane.
    The 10 fitted parameters are:
        params[0:3]  -> x-centers of 3 Gaussians
        params[3:6]  -> v-centers of 3 Gaussians
        params[6:9]  -> x-widths (sigma_x) of 3 Gaussians
        params[9]    -> shared v-width (sigma_v) for all Gaussians
    A Duffing-like base term provides smooth global oscillator dynamics.
    RBF amplitudes are fixed to keep BFGS well-conditioned.
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # Duffing-like base: -x - x^3 - damping*v captures global oscillator structure
    out = -1.0 * x - 0.5 * x * x * x - 0.3 * v

    # 3 anisotropic Gaussian RBF terms with fixed small amplitudes
    # Amplitudes chosen to be small enough not to dominate, large enough to add flexibility
    amp = 0.15
    for i in range(3):
        cx = p[i]
        cv = p[3 + i]
        sx = np.maximum(p[6 + i], 0.05)  # clip to avoid division by zero
        sv = np.maximum(p[9], 0.05)

        dx = (x - cx) / sx
        dv = (v - cv) / sv
        out = out + amp * np.exp(-0.5 * (dx * dx + dv * dv))

    return out
# EVOLVE-BLOCK-END
