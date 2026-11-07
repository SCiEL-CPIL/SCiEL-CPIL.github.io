"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a Ramberg-Osgood inspired formulation:
    sigma = E(T) * eps + K(T) * (eps * E(T))^(n-1) * E(T)
    
    Where:
    - E(T) = p0 + p1*T + p2*T^2  (elastic modulus as quadratic in temp)
    - K(T) = p3 + p4*T           (hardening coefficient as linear in temp)
    - n    = p5                  (hardening exponent)
    
    Actually, let's use a cleaner formulation:
    sigma = E(T) * eps + K(T) * (eps)^(n)
    
    But we need to be careful with negative strains. Let's use:
    sigma = E(T) * eps + K(T) * sign(eps) * |eps|^n
    
    Parameters:
    p0, p1, p2: coefficients for E(T) = p0 + p1*T + p2*T^2
    p3, p4:     coefficients for K(T) = p3 + p4*T
    p5:         hardening exponent n
    p6, p7, p8: additional flexibility terms
    p9:         offset
    
    Let's refine: 
    E(T) = p0 + p1*T
    K(T) = p2 + p3*T
    n = p4
    
    sigma = E(T)*eps + K(T)*sign(eps)*|eps|^n + p5*eps^2 + p6*eps^3 + p7*temp*eps + p8*temp^2*eps + p9*eps*temp^2
    """
    # Extract parameters
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
    
    # Elastic modulus as linear function of temperature
    E = p0 + p1 * temp
    
    # Hardening coefficient as linear function of temperature
    K = p2 + p3 * temp
    
    # Hardening exponent - keep it positive for stability
    n = np.abs(p4) + 1.0  # n >= 1 to ensure smooth behavior at origin
    
    # Base Ramberg-Osgood term
    # sigma_ro = E * eps + K * sign(eps) * |eps|^n
    abs_eps = np.abs(strain)
    sign_eps = np.sign(strain)
    # Handle zero strain carefully
    ro_term = E * strain + K * sign_eps * (abs_eps ** n)
    
    # Additional polynomial correction terms for better fitting
    # These capture deviations from pure Ramberg-Osgood
    correction = (
        p5 * strain**2 +
        p6 * strain**3 +
        p7 * temp * strain +
        p8 * temp**2 * strain +
        p9 * strain * temp**2
    )
    
    return ro_term + correction
# EVOLVE-BLOCK-END
