import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod using a rational function
    with temperature-dependent coefficients.
    Modified denominator to (1 + a3*e + a4*e^2) for better initial slope flexibility.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent base parameters
    a0 = p0 + p1 * temp
    a1 = p2 + p3 * temp
    a2 = p4 + p5 * temp
    a3 = p6 + p7 * temp
    a4 = p8 + p9 * temp
    
    # Rational function: sigma = (a0*e + a1*e^2 + a2*e^3) / (1 + a3*e + a4*e^2)
    # The linear term in the denominator allows the effective modulus to change with strain
    # more flexibly than the quadratic term alone.
    numerator = a0 * strain + a1 * strain**2 + a2 * strain**3
    denominator = 1.0 + a3 * strain + a4 * strain**2
    
    # Ensure denominator doesn't approach zero
    min_den = 1e-10
    sign_den = np.sign(denominator)
    sign_den = np.where(sign_den == 0, 1.0, sign_den)
    denominator = np.where(np.abs(denominator) < min_den, min_den * sign_den, denominator)
    
    stress = numerator / denominator
    
    return stress
