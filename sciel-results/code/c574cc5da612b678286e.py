"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Linear-in-parameters basis: polynomial + fixed-frequency Fourier terms
    # This makes the optimization a linear least-squares problem, so BFGS
    # converges reliably to the global optimum.
    #
    # params[0]: constant
    # params[1]: x
    # params[2]: x^2
    # params[3]: x^3
    # params[4]: x^4
    # params[5]: sin(1.0 * x)
    # params[6]: cos(1.0 * x)
    # params[7]: sin(2.0 * x)
    # params[8]: cos(2.0 * x)
    # params[9]: sin(3.0 * x)

    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2

    result = (
        params[0]
        + params[1] * x
        + params[2] * x2
        + params[3] * x3
        + params[4] * x4
        + params[5] * np.sin(x)
        + params[6] * np.cos(x)
        + params[7] * np.sin(2.0 * x)
        + params[8] * np.cos(2.0 * x)
        + params[9] * np.sin(3.0 * x)
    )

    return result
# EVOLVE-BLOCK-END
