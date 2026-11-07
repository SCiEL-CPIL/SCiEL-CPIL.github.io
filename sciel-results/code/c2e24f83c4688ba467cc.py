import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational stress-strain model for aluminium rod with polynomial corrections.

    Core: a rational saturation term that smoothly transitions from linear
    elastic (small strain) to a plateau + hardening (large strain), with
    temperature-dependent coefficients.

    Refinement: a low-order polynomial correction battery (strain^2, strain^3,
    temp*strain, temp^2, strain*temp^2) that absorbs systematic deviations the
    rational core cannot represent.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)

    # --- Temperature-dependent rational-core coefficients ---
    # Effective elastic modulus
    E = p0 + p1 * temp
    # Yield-like stress scale (numerator saturation level)
    S_y = p2 + p3 * temp
    # Transition sharpness (kept positive for numerical stability)
    a = np.abs(p4) + 1e-8
    # Post-transition hardening slope
    K = p5 + p6 * temp

    # --- Rational saturation core ---
    # stress_core = E*S_y*strain / (S_y + a*strain)
    #   small strain -> ~ E*strain          (elastic)
    #   large strain -> ~ E*S_y/a + K*strain^2 (plateau + hardening)
    denom = S_y + a * strain
    denom_safe = np.where(np.abs(denom) < 1e-12, np.where(denom >= 0, 1e-12, -1e-12), denom)
    stress_core = (E * S_y * strain) / denom_safe + K * strain**2

    # --- Polynomial correction battery (refinement machinery) ---
    stress = stress_core + (
        p7 * strain**2
        + p8 * temp * strain
        + p9 * temp**2
    )

    return stress
