"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Combines tanh-saturation hardening with polynomial corrections.
    """
    e = strain
    t = temp

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Core elastic-plastic response with saturation
    elastic = p1 * e
    hardening = p3 * np.tanh(p2 * np.maximum(e, 0.0))

    # Temperature corrections
    temp_effect = p4 * (t - 0.5)
    temp_yield = p5 * (t - 0.5) * e
    temp_hard = p6 * (t - 0.5) * np.tanh(p2 * np.maximum(e, 0.0))

    # Higher-order corrections for nonlinearity
    quad = p7 * e**2 * (1.0 - np.tanh(p2 * np.maximum(e, 0.0)))
    cubic = p8 * e**3 * (1.0 - np.tanh(p2 * np.maximum(e, 0.0)))
    cross = p9 * t * e**2

    return (p0 + elastic + hardening + temp_effect + temp_yield +
            temp_hard + quad + cubic + cross)
# EVOLVE-BLOCK-END
