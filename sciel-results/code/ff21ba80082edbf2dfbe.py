import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod using a rational function form.
    
    Structure:
    stress = (a + b*strain + c*strain^2 + d*strain^3) / (1 + e*strain + f*strain^2) + (g + h*temp) * strain
    
    The rational part models the elastic-plastic transition (knee) more efficiently
    than polynomials, while the linear term captures the overall trend with temperature.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Numerator of rational part
    num = p0 + p1 * strain + p2 * strain**2 + p3 * strain**3
    
    # Denominator of rational part
    den = 1.0 + p4 * strain + p5 * strain**2
    
    # Avoid division by zero
    den_safe = np.where(np.abs(den) < 1e-10, np.sign(den) * 1e-10 + 1e-10, den)
    
    rational_part = num / den_safe
    
    # Linear temperature-dependent term
    linear_part = (p6 + p7 * temp) * strain
    
    # Constant offset
    stress = rational_part + linear_part + p8 + p9 * temp
    
    return stress
