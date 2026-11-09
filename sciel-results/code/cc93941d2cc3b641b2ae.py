"""Improved model: piecewise-linear + smooth saturation for aluminium stress-strain."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Combines piecewise-linear (min/max) basis for yield points with
    smooth tanh-based saturation for work-hardening, plus temperature
    dependence for elastic modulus and yield strength.
    
    params (10 constants):
      p0: Base elastic modulus
      p1: Temperature coefficient for E
      p2: Yield strain scale
      p3: Yield stress scale
      p4: Hardening rate
      p5: Saturation strain scale (tanh width)
      p6: Second yield point location
      p7: Second yield amplitude
      p8: Linear hardening slope
      p9: Offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Temperature-dependent elastic modulus (linear in T)
    E_T = p0 + p1 * T
    
    # Temperature-dependent yield scale
    Y_T = p3 * (1.0 - 0.02 * T)  # slight temp reduction in yield
    
    # Piecewise-linear yield feature using min/max (differentiable a.e.)
    # First yield: ramps from 0 to 1 over [0, p2]
    yield1 = np.minimum(eps / np.where(np.abs(p2) > 1e-8, p2, 1.0), 1.0)
    yield1 = np.maximum(yield1, 0.0)
    
    # Second yield feature: ramps from 0 to 1 over [p6, p6+p2]
    yield2 = np.minimum((eps - p6) / np.where(np.abs(p2) > 1e-8, p2, 1.0), 1.0)
    yield2 = np.maximum(yield2, 0.0)
    
    # Smooth saturation via tanh: models work-hardening plateau
    sat_scale = np.where(np.abs(p5) > 1e-8, p5, 1.0)
    saturation = np.tanh(eps / sat_scale)
    
    # Combined plastic shape
    plastic = (Y_T * yield1 + p7 * yield2 + p4 * saturation + p8 * eps)
    
    # Elastic + plastic
    sigma = E_T * eps + plastic + p9
    
    # Guard against non-finite values
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    return sigma
# EVOLVE-BLOCK-END
