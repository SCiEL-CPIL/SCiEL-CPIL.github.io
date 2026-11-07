"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = params
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    return (p[0] + p[1] * x + p[2] * x2 + p[3] * x3 + p[4] * x4 + p[5] * x5
            + p[6] * np.sin(x) + p[7] * np.cos(x)
            + p[8] * np.exp(-0.5 * x2) + p[9] * np.tanh(x))
# EVOLVE-BLOCK-END
