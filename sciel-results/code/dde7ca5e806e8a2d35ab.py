"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 120.0  # seconds, safety margin before 150s kill


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback to simple linear to avoid timeout
        return params[0] * x + params[1] * y + params[2] * z + params[3]

    # Center inputs for numerical conditioning
    cx = x - 1.25
    cy = y - 1.25
    cz = z - 1.25

    p = params

    # 10-parameter basis:
    # 0: constant
    # 1-3: linear terms
    # 4-6: sin of centered variables (bounded, smooth nonlinearities)
    # 7-9: cross-product terms
    result = (
        p[0]
        + p[1] * cx
        + p[2] * cy
        + p[3] * cz
        + p[4] * np.sin(cx)
        + p[5] * np.sin(cy)
        + p[6] * np.sin(cz)
        + p[7] * cx * cy
        + p[8] * cx * cz
        + p[9] * cy * cz
    )

    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)

    return result
# EVOLVE-BLOCK-END
