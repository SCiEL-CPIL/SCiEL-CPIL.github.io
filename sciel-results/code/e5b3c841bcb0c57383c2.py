"""Symbolic regression: E. coli growth rate using Monod + Fourier basis."""
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    Structure:
      - Monod kinetics for substrate saturation
      - Fourier series for temperature and pH (non-monotonic optimal ranges)
      - Logistic density inhibition
      - Baseline offset
    
    Parameters (10):
      p0: Monod Ks (substrate half-saturation)
      p1: Monod amplitude (max substrate-limited growth)
      p2: temperature frequency (rad per unit temp)
      p3: temperature amplitude
      p4: temperature phase
      p5: pH frequency (rad per unit pH)
      p6: pH amplitude
      p7: pH phase
      p8: density inhibition coefficient
      p9: baseline offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Ensure Ks is positive for numerical stability
    Ks = np.abs(p0) + 1e-6

    # Monod term: substrate saturation
    monod = p1 * (s / (Ks + s))

    # Fourier terms for temperature (centered around 37C, typical E. coli optimum)
    temp_centered = temp - 37.0
    temp_fourier = p3 * np.sin(p2 * temp_centered + p4)

    # Fourier terms for pH (centered around 7.0, typical E. coli optimum)
    ph_centered = pH - 7.0
    ph_fourier = p6 * np.sin(p5 * ph_centered + p7)

    # Density-dependent inhibition: reduces growth at high population density
    # Use a smooth inhibition: 1 / (1 + p8 * b)
    density_factor = 1.0 / (1.0 + np.abs(p8) * b)

    # Combine: growth = baseline + (monod + temp_fourier + ph_fourier) * density_factor
    # The additive terms are gated by density inhibition
    growth = p9 + (monod + temp_fourier + ph_fourier) * density_factor

    # Ensure non-negative growth rate (biological constraint)
    growth = np.maximum(growth, 0.0)

    return growth
