"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Approximate target function using a small tanh-based model with 10 params."""
    # 3 hidden units with tanh activation
    h1 = np.tanh(params[0] * x + params[3])
    h2 = np.tanh(params[1] * x + params[4])
    h3 = np.tanh(params[2] * x + params[5])
    # Output layer
    return params[6] * h1 + params[7] * h2 + params[8] * h3 + params[9]
# EVOLVE-BLOCK-END
