"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = params
    u = x / 3.0
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    exp_neg_u2 = np.exp(-0.5 * u2)
    return (p[0] + p[1] * u + p[2] * u2 + p[3] * u3 + p[4] * u4
            + p[5] * np.sin(u) + p[6] * np.cos(u)
            + p[7] * exp_neg_u2 + p[8] * np.tanh(u)
            + p[9] * (u * exp_neg_u2))
# EVOLVE-BLOCK-END
