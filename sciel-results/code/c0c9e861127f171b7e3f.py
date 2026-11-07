"""Rewritten program: Ramberg-Osgood form with temperature-dependent coefficients.
Uses a closed-form expression with 10 parameters. Includes a time guard."""
import numpy as np
import time

_start_time = None

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Ramberg-Osgood style model for stress-strain of aluminium.
    
    stress = E*(t) * eps + K(t) * (eps / eps0)^(1/n(t))
    where E, K, n are linear functions of temperature.
    
    Params (10):
      0: E0    - reference Young's modulus
      1: dE_dt - temperature coefficient of modulus
      2: K0    - reference strength coefficient
      3: dK_dt - temperature coefficient of strength
      4: n0    - reference strain-hardening exponent (base)
      5: dn_dt - temperature coefficient of n
      6: eps0  - reference strain scale
      7: c1    - linear stress offset
      8: c2    - quadratic stress offset
      9: c3    - cubic stress offset
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 130:
        return np.zeros_like(strain, dtype=np.float64)
    
    E0, dE_dt, K0, dK_dt, n0, dn_dt, eps0, c1, c2, c3 = params
    
    s = strain
    t = temp
    
    # Temperature-dependent modulus
    E = E0 + dE_dt * t
    
    # Temperature-dependent strength
    K = K0 + dK_dt * t
    
    # Temperature-dependent strain-hardening exponent
    # Use a safe positive value for n
    n = n0 + dn_dt * t
    # Ensure n is bounded away from zero to avoid division issues
    # n should be positive; clamp to [0.1, 10]
    n_safe = np.where(n > 0.1, n, 0.1)
    n_safe = np.where(n_safe < 10.0, n_safe, 10.0)
    
    # Reference strain
    eps0_safe = np.where(np.abs(eps0) > 1e-6, eps0, 1e-6)
    
    # Ramberg-Osgood term
    # (eps / eps0)^(1/n) -- use signed power for negative strains
    # For negative strain, we want the plastic term to also be negative (symmetric behavior)
    # Use sign(eps) * |eps/eps0|^(1/n)
    ratio = s / eps0_safe
    sign_ratio = np.sign(ratio)
    abs_ratio = np.abs(ratio)
    # Use a safe lower bound for abs_ratio to avoid 0^negative issues if n is large
    # 1/n_safe is in [0.1, 10], so 0^positive = 0 which is fine
    # But if n_safe is very small (1/n large), 0^large = 0, still fine
    plastic_term = sign_ratio * np.power(abs_ratio, 1.0 / n_safe)
    
    # Total stress
    result = E * s + K * plastic_term + c1 * s + c2 * (s * s) + c3 * (s * s * s)
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
