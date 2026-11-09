"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a rational function form (Padé-like) to capture complex non-linearities:
    stress = (a(t)*s + b(t)*s^2) / (1 + c(t)*s + d(t)*s^2) + e*s + f
    
    This allows for:
    - Linear elastic response (initial slope)
    - Yield plateau or sharp transition (inflection in rational part)
    - Hardening/Saturation (asymptotic behavior)
    - Full temperature dependence in all coefficients
    
    Parameters:
    p0, p1: a(t) = p0 + p1*t
    p2, p3: b(t) = p2 + p3*t
    p4, p5: c(t) = p4 + p5*t
    p6, p7: d(t) = p6 + p7*t
    p8:     constant offset f
    p9:     linear slope correction e
    """
    s = np.asarray(strain, dtype=np.float64)
    t = np.asarray(temp, dtype=np.float64)
    
    # Ensure strain is non-negative for physical consistency, though rational handles negative
    # but stress-strain is usually plotted for s>=0. The data range includes negative strain.
    # We should handle negative strain. Rational functions are fine for negative s.
    
    p0, p1, p2, p3, p4, p5, p6, p7, f_offset, e_slope = params
    
    # Temperature-dependent coefficients
    a_t = p0 + p1 * t
    b_t = p2 + p3 * t
    c_t = p4 + p5 * t
    d_t = p6 + p7 * t
    
    # Calculate numerator and denominator
    # s^2 can be large, but s is in [-0.001, 0.96]. s^2 is small.
    s2 = s * s
    
    numerator = a_t * s + b_t * s2
    denominator = 1.0 + c_t * s + d_t * s2
    
    # Avoid division by zero or near-zero
    # If denominator is close to 0, the function blows up. 
    # We add a small epsilon to denominator to prevent inf/nan during optimization.
    # However, BFGS might step into bad regions. 
    # A safer approach for rational functions in optimization is to use a soft denominator or just rely on the optimizer staying in valid regions.
    # Given the range of s is small, and initial params are 1.0:
    # Denom = 1 + 1*s + 1*s2. For s in [-0.002, 0.96], Denom is approx [0.99, 2.9]. Safe.
    # If optimizer moves c, d to large values, we might hit zeros.
    # Let's clip the denominator to avoid singularities.
    denom_safe = np.where(np.abs(denominator) < 1e-6, 1e-6 * np.sign(denominator), denominator)
    # Actually, sign(0) is 0. Let's just use a floor on absolute value and preserve sign?
    # Simpler: just add epsilon.
    denom_safe = denominator + 1e-8 * np.sign(denominator)
    # If denominator is exactly 0, sign is 0, so we add 0. Let's use:
    denom_safe = np.where(denominator == 0, 1e-6, denominator)
    
    rational_part = numerator / denom_safe
    
    # Add linear correction and offset
    stress = rational_part + e_slope * s + f_offset
    
    return stress
# EVOLVE-BLOCK-END
