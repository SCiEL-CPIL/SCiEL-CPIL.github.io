"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 10-parameter basis: richer set including cross-terms, higher-order polys, and multiple frequencies
    # Designed to capture polynomial, trigonometric, and interaction effects
    return (
        params[0] * x
        + params[1] * y
        + params[2] * x * y
        + params[3] * x * x
        + params[4] * y * y
        + params[5] * np.sin(np.pi * x)
        + params[6] * np.sin(np.pi * y)
        + params[7] * np.sin(2.0 * np.pi * x)
        + params[8] * np.sin(2.0 * np.pi * y)
        + params[9]
    )
# EVOLVE-BLOCK-END
