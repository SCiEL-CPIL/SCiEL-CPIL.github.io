"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s limit

    # Fit full degree-5 polynomial
    try:
        coeff = np.polynomial.polynomial.polyfit(
            training_x, training_subgrid_tendency, 5
        )
        # Validate: all coefficients must be finite
        if not np.all(np.isfinite(coeff)):
            raise ValueError("Non-finite coefficients")
        return coeff
    except Exception:
        pass

    # Fallback: degree-3 if degree-5 fails
    try:
        coeff = np.zeros(6)
        coeff[:4] = np.polynomial.polynomial.polyfit(
            training_x, training_subgrid_tendency, 3
        )
        if np.all(np.isfinite(coeff)):
            return coeff
    except Exception:
        pass

    # Final fallback: linear
    try:
        coeff = np.zeros(6)
        coeff[:2] = np.polynomial.polynomial.polyfit(
            training_x, training_subgrid_tendency, 1
        )
        if np.all(np.isfinite(coeff)):
            return coeff
    except Exception:
        pass

    # Absolute fallback: zeros
    return np.zeros(6)
# EVOLVE-BLOCK-END
