"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod.
    
    Uses a rational function form that can capture:
    - Linear elastic regime at low strain
    - Plastic hardening and saturation
    - Temperature effects (softening)
    
    params length 10:
    p0: elastic modulus (slope in linear regime)
    p1: plastic saturation stress
    p2: strain scale for transition (elastic to plastic)
    p3: temperature coefficient (linear softening)
    p4: temperature quadratic coefficient
    p5: yield point offset
    p6: hardening exponent (shape parameter)
    p7: temp-strain interaction
    p8: denominator shift (rational flexibility)
    p9: plastic strain scale (for saturation)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Safeguard against division by zero and negative scales
    p2_safe = np.maximum(abs(p2), 1e-8)
    p9_safe = np.maximum(abs(p9), 1e-8)
    
    # Positive plastic strain (yielding only in tension)
    eps_plastic = np.maximum(strain, 0.0)
    
    # Elastic component: linear with strain
    elastic = p0 * strain
    
    # Plastic hardening: rational saturation form
    # As eps_plastic -> 0, hardening -> 0
    # As eps_plastic -> inf, hardening -> p1
    # The power p6 controls the rate of approach to saturation
    ratio = eps_plastic / (eps_plastic + p9_safe)
    hardening = p1 * np.power(ratio, p6)
    
    # Temperature effects: thermal softening
    # Linear and quadratic in temperature
    temp_softening = p3 * temp + p4 * temp**2
    
    # Temperature-strain interaction: thermal effects on plastic flow
    temp_interaction = p7 * temp * eps_plastic
    
    # Yield offset: shift for the onset of plasticity
    yield_offset = p5
    
    # Numerator: sum of all stress contributions
    numerator = elastic + hardening + temp_softening + temp_interaction + yield_offset
    
    # Denominator: rational function flexibility
    # This allows the model to capture more complex stress-strain behavior
    # At low strain, denominator ~ 1, so stress ~ numerator
    # At high strain, denominator grows, potentially causing softening
    denominator = 1.0 + p8 * eps_plastic
    
    # Final stress prediction
    stress = numerator / denominator
    
    return stress
# EVOLVE-BLOCK-END
