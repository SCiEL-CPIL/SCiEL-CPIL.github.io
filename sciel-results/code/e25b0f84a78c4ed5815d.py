"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled as a 4th-order multivariate polynomial in normalized inputs.
    
    Inputs:
        b: population density
        s: substrate concentration
        temp: temperature
        pH: pH level
        params: array of 10 fitted constants
    
    The inputs are normalized to approximately [0, 1] based on training ranges:
        b: [1, 4.998] -> (b - 1) / 3.998
        s: [1, 4.999] -> (s - 1) / 3.999
        temp: [1.002, 39.98] -> (temp - 1.002) / 38.978
        pH: [2, 12] -> (pH - 2) / 10
    
    A 4th-order polynomial with 10 terms captures main effects, quadratic
    curvature, and key cross-variable interactions.
    """
    # Normalize inputs to [0, 1] range
    b_norm = (b - 1.0) / 3.998
    s_norm = (s - 1.0) / 3.999
    temp_norm = (temp - 1.002) / 38.978
    ph_norm = (pH - 2.0) / 10.0

    # Clip to avoid extreme values outside training range
    b_norm = np.clip(b_norm, 0.0, 1.0)
    s_norm = np.clip(s_norm, 0.0, 1.0)
    temp_norm = np.clip(temp_norm, 0.0, 1.0)
    ph_norm = np.clip(ph_norm, 0.0, 1.0)

    # 10 terms:
    # 1. constant
    # 2. b
    # 3. s
    # 4. temp
    # 5. pH
    # 6. temp^2   (optimal temperature)
    # 7. pH^2     (optimal pH)
    # 8. b * s    (density-substrate interaction)
    # 9. s * temp (substrate-temperature interaction)
    # 10. b * pH  (density-pH interaction)
    
    t1 = np.ones_like(b_norm)
    t2 = b_norm
    t3 = s_norm
    t4 = temp_norm
    t5 = ph_norm
    t6 = temp_norm * temp_norm
    t7 = ph_norm * ph_norm
    t8 = b_norm * s_norm
    t9 = s_norm * temp_norm
    t10 = b_norm * ph_norm

    prediction = (
        params[0] * t1 +
        params[1] * t2 +
        params[2] * t3 +
        params[3] * t4 +
        params[4] * t5 +
        params[5] * t6 +
        params[6] * t7 +
        params[7] * t8 +
        params[8] * t9 +
        params[9] * t10
    )

    return prediction
# EVOLVE-BLOCK-END
