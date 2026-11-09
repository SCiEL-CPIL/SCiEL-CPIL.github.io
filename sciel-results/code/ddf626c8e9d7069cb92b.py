# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Uses a smooth rational basis with temperature-dependent coefficients.
    The form combines:
    - A linear elastic term with temperature-dependent modulus
    - A rational function for the yield-to-plastic transition
    - A saturation term for work hardening
    All terms are smooth and differentiable for BFGS.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to [0, 1] range for better conditioning
    s = (strain + 0.002) / 0.966
    t = (temp - 0.06667) / 0.93333
    
    # Temperature-dependent elastic modulus
    E = p0 * (1.0 + p1 * t)
    
    # Temperature-dependent yield stress
    sigma_y = p2 * (1.0 - p3 * t)
    
    # Smooth yield transition using rational function
    # This creates a smooth S-curve from elastic to plastic
    # The denominator ensures smoothness and controls transition width
    width = np.maximum(np.abs(p4), 0.01)
    # Transition variable: how far into plastic regime
    x = (s * E - sigma_y) / width
    
    # Smooth transition: 0 in elastic, 1 in plastic
    # Using a rational function instead of tanh for better conditioning
    trans = x**2 / (1.0 + x**2)
    
    # Elastic stress
    stress_elastic = E * s
    
    # Plastic stress with work hardening
    # Use a combination of power law and rational saturation
    eps_plastic = np.maximum(s - sigma_y / np.maximum(np.abs(E), 1e-8), 0.0)
    
    # Work hardening: power law with temperature dependence
    hardening_power = np.maximum(np.abs(p5), 0.05)
    hardening = p6 * (1.0 - p7 * t) * eps_plastic**hardening_power
    
    # Saturation term to prevent unbounded growth
    # Rational saturation: approaches a maximum as plastic strain increases
    sat_scale = np.maximum(np.abs(p8), 0.1)
    sat_width = np.maximum(np.abs(p9), 0.05)
    saturation = sat_scale * eps_plastic / (sat_width + eps_plastic)
    
    # Plastic stress = yield + hardening + saturation
    stress_plastic = sigma_y + hardening + saturation
    
    # Blend elastic and plastic using smooth transition
    stress = (1.0 - trans) * stress_elastic + trans * stress_plastic
    
    # Ensure finite output
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
