"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a nonlinear physical model: 
    Stress = (E(T) * strain) * (1 + hardening * strain^2) + offset
    
    Where E(T) is a temperature-dependent Young's modulus:
    E(T) = p0 * (1 - p1 * (T - T_ref))
    
    This structure captures:
    1. Linear elastic regime (Hooke's law)
    2. Temperature dependence of stiffness (aluminium softens with heat)
    3. Nonlinear hardening/softening (plastic regime approximation)
    4. Constant offset for zero-strain stress
    
    Parameters (10 total):
    p[0]: Base Young's modulus (E0)
    p[1]: Temperature coefficient of E
    p[2]: T_ref (reference temperature, centered)
    p[3]: Hardening coefficient (quadratic strain term)
    p[4]: Cubic strain correction
    p[5]: Temperature-strain interaction in hardening
    p[6]: Offset
    p[7]: Linear temperature stress shift
    p[8]: Quadratic temperature term
    p[9]: Higher order temp-strain interaction
    """
    # Extract parameters
    E0 = params[0]
    dE_dT = params[1]
    T_ref = params[2]
    harden = params[3]
    cubic_corr = params[4]
    temp_harden_inter = params[5]
    offset = params[6]
    temp_linear_shift = params[7]
    temp_quad = params[8]
    temp_strain_high = params[9]

    # Center temperature for numerical stability
    T_c = temp - 0.5
    
    # Calculate temperature-dependent modulus
    # E(T) = E0 * (1 - dE_dT * (T_c - T_ref))
    # We use T_ref as a parameter to allow the fit to find the best reference point
    E_T = E0 * (1.0 - dE_dT * (T_c - T_ref))
    
    # Base linear elastic stress
    stress_linear = E_T * strain
    
    # Nonlinear hardening terms
    # The term (1 + harden * strain^2 + cubic_corr * strain^3) approximates plastic behavior
    # We add a temperature-dependent modification to the hardening
    hardening_factor = 1.0 + harden * (strain * strain) + cubic_corr * (strain * strain * strain)
    # Add temperature interaction to hardening: as temp increases, hardening might decrease
    hardening_factor = hardening_factor + temp_harden_inter * T_c * (strain * strain)
    
    stress_nonlinear = stress_linear * hardening_factor
    
    # Temperature-induced stress shifts (thermal expansion effects or measurement offsets)
    # These are added to the mechanical stress
    temp_stress = temp_linear_shift * T_c + temp_quad * (T_c * T_c)
    
    # Higher order interaction
    interaction = temp_strain_high * T_c * strain
    
    # Total stress
    stress = stress_nonlinear + temp_stress + interaction + offset
    
    return stress
# EVOLVE-BLOCK-END
