"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 10-parameter basis: mix of polynomial and trigonometric terms
    # This covers both smooth polynomial-like and oscillatory target functions
    return (
        params[0] * x
        + params[1] * y
        + params[2] * x * y
        + params[3] * np.sin(np.pi * x)
        + params[4] * np.sin(np.pi * y)
        + params[5] * np.cos(np.pi * x)
        + params[6] * np.cos(np.pi * y)
        + params[7] * x * x
        + params[8] * y * y
        + params[9]
    )
# EVOLVE-BLOCK-END
