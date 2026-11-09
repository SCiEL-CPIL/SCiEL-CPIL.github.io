"""Bootstrap portfolio: model-and-optimize for stress-strain of aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT_LIMIT = 140.0

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.

    Design family: Fourier series basis in strain with temperature-dependent
    coefficients to model complex elastic-plastic transitions and work-hardening.

    Model:
      stress = sum_{k=0}^{K} (a_k + b_k * temp) * cos(k * omega * strain)
             + sum_{k=1}^{K} (c_k + d_k * temp) * sin(k * omega * strain)

    With 10 parameters, we use:
      p0: omega (fundamental frequency)
      p1: a_1 (cos coefficient, order 1)
      p2: b_1 (cos coefficient temp dep, order 1)
      p3: c_1 (sin coefficient, order 1)
      p4: d_1 (sin coefficient temp dep, order 1)
      p5: a_2 (cos coefficient, order 2)
      p6: b_2 (cos coefficient temp dep, order 2)
      p7: c_2 (sin coefficient, order 2)
      p8: d_2 (sin coefficient temp dep, order 2)
      p9: offset (constant bias)

    This captures periodic-like curvature patterns and inflection points inherent
    in aluminium's stress-strain behavior through the Fourier basis.
    """
    if time.time() - _START_TIME > _TIMEOUT_LIMIT:
        return np.zeros_like(strain, dtype=np.float64)

    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)

    omega = p0
    a1, b1, c1, d1 = p1, p2, p3, p4
    a2, b2, c2, d2 = p5, p6, p7, p8
    offset = p9

    # Compute angular arguments
    theta1 = omega * strain
    theta2 = 2.0 * omega * strain

    # Temperature-dependent coefficients
    cos_coeff_1 = a1 + b1 * temp
    sin_coeff_1 = c1 + d1 * temp
    cos_coeff_2 = a2 + b2 * temp
    sin_coeff_2 = c2 + d2 * temp

    # Fourier series terms
    stress = (
        cos_coeff_1 * np.cos(theta1) +
        sin_coeff_1 * np.sin(theta1) +
        cos_coeff_2 * np.cos(theta2) +
        sin_coeff_2 * np.sin(theta2) +
        offset
    )

    # Ensure finite output
    stress = np.where(np.isfinite(stress), stress, 0.0)

    return stress
# EVOLVE-BLOCK-END
