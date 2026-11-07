import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Model: Ramberg-Osgood with temperature-dependent coefficients.
    sigma = E(T)*eps + K(T)*|eps|^n(T)*sign(eps) + Q(T)*eps^2 + C(T)
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent coefficients
    # E(T): Elastic Modulus
    E = p0 + p1 * temp
    # K(T): Hardening Coefficient
    K = p2 + p3 * temp
    # n(T): Hardening Exponent (clamped to avoid numerical issues with negative or very small exponents)
    n = np.maximum(p4 + p5 * temp, 0.1)
    # Q(T): Quadratic correction
    Q = p6 + p7 * temp
    # C(T): Constant offset
    C = p8 + p9 * temp
    
    # Calculate terms
    # Elastic term
    term_elastic = E * strain
    
    # Plastic/Hardening term
    # Use sign(eps) * |eps|^n
    # np.sign(0) is 0, which is fine.
    term_hardening = K * np.power(np.abs(strain), n) * np.sign(strain)
    
    # Quadratic term
    term_quadratic = Q * strain**2
    
    # Constant term
    term_const = C
    
    return term_elastic + term_hardening + term_quadratic + term_const
