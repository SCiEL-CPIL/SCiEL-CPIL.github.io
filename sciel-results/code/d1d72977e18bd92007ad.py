# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(params, dtype=float)

    # 10 basis functions, one per free parameter. A mix of polynomial,
    # trigonometric and saturation terms so the 10 coefficients can adapt
    # to a wide range of target shapes while staying linear in params
    # (fast, well-conditioned, BFGS-friendly).
    B = np.empty((x.size, 10), dtype=float)
    B[:, 0] = 1.0                      # constant
    B[:, 1] = x                        # linear x
    B[:, 2] = y                        # linear y
    B[:, 3] = x * y                    # interaction
    B[:, 4] = x * x - 0.5              # quadratic x (centered)
    B[:, 5] = y * y - 0.5              # quadratic y (centered)
    B[:, 6] = np.tanh(x * y)           # saturation interaction
    B[:, 7] = np.cos(np.pi * x * y)    # periodic interaction
    B[:, 8] = np.tanh(x + y)           # saturation sum
    B[:, 9] = np.tanh(x - y)           # saturation difference

    return B @ p
# EVOLVE-BLOCK-END
