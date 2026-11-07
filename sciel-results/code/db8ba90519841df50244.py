"""Stress-strain of an aluminium rod: elastic-plastic with yield offset and hardening."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Elastic-plastic model with yield offset:
    
    stress = E(T)*e + H(T)*max(e - eps_y(T), 0)^n(T) + K(T)*exp(-e/alpha) + c0 + c1*T
    
    where:
      e = strain (clipped to positive for power terms)
      E(T) = p0 + p1*T           (elastic modulus)
      eps_y(T) = p2 + p3*T       (yield strain, thermal softening)
      H(T) = p4 + p5*T           (hardening magnitude)
      n(T) = p6 + p7*T           (hardening exponent)
      K(T) = p8                  (transient/exponential term magnitude)
      alpha = 1e-2               (fixed decay scale)
      c0, c1 = baseline terms
    
    All 10 params used. Smooth, closed-form, BFGS-friendly.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, c1 = params

    # Ensure strain is non-negative for power operations
    e = np.clip(strain, 1.0e-8, None)

    # Temperature-dependent coefficients
    E = p0 + p1 * temp                    # elastic modulus
    eps_y = p2 + p3 * temp                # yield strain
    H = p4 + p5 * temp                    # hardening magnitude
    n = np.clip(p6 + p7 * temp, 0.1, 5.0) # hardening exponent (clamped for stability)

    # Elastic contribution
    elastic = E * e

    # Plastic hardening: only active beyond yield
    plastic_strain = np.clip(e - eps_y, 0.0, None)
    plastic = H * np.power(plastic_strain, n)

    # Exponential transient (sharp initial rise)
    transient = p8 * np.exp(-e / 1.0e-2)

    # Temperature baseline
    base = c1 * temp

    stress = elastic + plastic + transient + base
    return stress
# EVOLVE-BLOCK-END
