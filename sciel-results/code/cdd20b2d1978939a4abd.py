"""E. coli growth rate model with mechanistic Monod + Gaussian temperature/pH response."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_deadline_start = time.time()
_DEADLINE = 140.0  # Safety margin before 150s wall-clock limit


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate with mechanistic basis functions.
    
    Uses Monod substrate saturation, Gaussian temperature and pH optima,
    and rational density inhibition. This creates a smooth, well-conditioned
    landscape for BFGS optimization.
    
    Basis functions:
      1. Core: Monod * temp_gauss * ph_gauss * density_inhib
      2-4. Individual component terms
      5-7. Pairwise interactions
      8-9. Correction terms
      10. Constant offset
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Fixed biological parameters (E. coli physiology)
    T_opt = 37.0      # Optimal temperature (°C)
    T_sigma = 8.0     # Temperature width
    pH_opt = 7.0      # Optimal pH
    pH_sigma = 1.5    # pH width
    Ks = 3.0          # Substrate half-saturation
    Kd = 0.5          # Density inhibition constant

    # Deadline guard
    if time.time() - _deadline_start > _DEADLINE:
        return np.zeros_like(b)

    # Gaussian temperature response
    temp_gauss = np.exp(-((temp - T_opt) ** 2) / (2.0 * T_sigma ** 2))

    # Gaussian pH response
    ph_gauss = np.exp(-((pH - pH_opt) ** 2) / (2.0 * pH_sigma ** 2))

    # Monod substrate saturation
    monod_s = s / (Ks + s)

    # Rational density inhibition (Haldane-type)
    density_inhib = 1.0 / (1.0 + Kd * b + 0.1 * b * b)

    # Core multiplicative term
    core = monod_s * temp_gauss * ph_gauss * density_inhib

    # Individual component terms
    temp_only = temp_gauss
    ph_only = ph_gauss
    substrate_only = monod_s

    # Pairwise interactions
    temp_ph = temp_gauss * ph_gauss
    substrate_temp = monod_s * temp_gauss
    substrate_ph = monod_s * ph_gauss

    # Correction terms
    density_sub = density_inhib * monod_s
    temp_density = temp_gauss * density_inhib

    # Combine all basis functions with fitted parameters
    prediction = (
        p0 * core +
        p1 * temp_only +
        p2 * ph_only +
        p3 * substrate_only +
        p4 * temp_ph +
        p5 * substrate_temp +
        p6 * substrate_ph +
        p7 * density_sub +
        p8 * temp_density +
        p9
    )

    return prediction
# EVOLVE-BLOCK-END
