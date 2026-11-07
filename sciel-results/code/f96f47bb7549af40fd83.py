"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a physically-motivated model for aluminium stress-strain behavior:
    - Elastic region (linear Hooke's law)
    - Yield plateau (slight hardening)
    - Strain hardening region (power-law)
    - Temperature effects on yield strength and elastic modulus
    
    The model uses a smooth transition between regions using sigmoid functions
    to avoid discontinuities that would hinder BFGS optimization.
    
    params (10 constants):
    p0: Elastic modulus E (baseline)
    p1: Yield strain offset
    p2: Yield stress level
    p3: Hardening exponent n
    p4: Hardening coefficient K
    p5: Temperature coefficient for E
    p6: Temperature coefficient for yield
    p7: Transition sharpness for elastic-to-plastic
    p8: Transition sharpness for plastic-to-hardening
    p9: Additional offset/intercept
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    e = strain
    t = temp
    
    # Temperature-dependent elastic modulus and yield properties
    E_temp = p0 * (1.0 + p5 * (t - 0.5))
    yield_strain = p1 + p6 * (t - 0.5)
    yield_stress = p2 * (1.0 + p6 * (t - 0.5))
    
    # Ensure physical constraints via softplus-like transformations
    # This keeps parameters in reasonable ranges during optimization
    E_safe = np.maximum(E_temp, 1e-6)
    yield_stress_safe = np.abs(yield_stress) + 1e-6
    
    # Smooth transition from elastic to plastic
    # For e < yield_strain: elastic stress = E * e
    # For e > yield_strain: plastic stress with hardening
    
    # Use sigmoid to create smooth transition
    # alpha controls sharpness of transition
    alpha = np.maximum(np.abs(p7), 0.1)
    beta = np.maximum(np.abs(p8), 0.1)
    
    # Normalize strain relative to yield point
    e_rel = e - yield_strain
    
    # Elastic stress
    stress_elastic = E_safe * e
    
    # Plastic stress: yield_stress + K * (e - yield_strain)^n
    # Only for positive plastic strain
    e_plastic = np.maximum(e_rel, 0.0)
    # Handle negative n or zero strain carefully
    e_plastic_safe = np.maximum(e_plastic, 1e-10)
    n_safe = np.maximum(np.abs(p3), 0.01)
    K_safe = np.maximum(np.abs(p4), 1e-6)
    
    stress_plastic = yield_stress_safe + K_safe * np.power(e_plastic_safe, n_safe)
    
    # Smooth blend between elastic and plastic
    # sigmoid(alpha * e_rel) is 0 when e << yield_strain, 1 when e >> yield_strain
    blend = 1.0 / (1.0 + np.exp(-alpha * e_rel))
    
    # Combine: elastic where blend~0, plastic where blend~1
    stress = (1.0 - blend) * stress_elastic + blend * stress_plastic
    
    # Add small intercept
    stress = stress + p9
    
    # For negative strain (compression), use symmetric elastic behavior
    # but with slightly different modulus if needed
    negative_mask = e < 0
    if np.any(negative_mask):
        stress_neg = E_safe * e + p9
        stress = np.where(negative_mask, stress_neg, stress)
    
    return stress
# EVOLVE-BLOCK-END
