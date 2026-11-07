import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a Padé-like rational function model with temperature-dependent coefficients.
    This form provides a smooth optimization landscape and naturally captures
    the saturation and hardening behavior of metals.
    
    Model:
        sigma = (N(strain, T)) / (D(strain, T)) + Offset(T)
        
        N = (p0 + p1*T)*strain + (p2 + p3*T)*strain^2
        D = 1 + (p4 + p5*T)*strain + (p6 + p7*T)*strain^2
        Offset = p8 + p9*T
        
    The rational term allows for flexible curvature, transitioning from
    linear elastic behavior (if D ~ 1) to saturation (if D grows).
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent coefficients
    # Numerator terms
    a1 = p0 + p1 * temp
    a2 = p2 + p3 * temp
    
    # Denominator terms
    b1 = p4 + p5 * temp
    b2 = p6 + p7 * temp
    
    # Offset
    offset = p8 + p9 * temp
    
    # Compute rational part
    # N = a1 * strain + a2 * strain^2
    # D = 1 + b1 * strain + b2 * strain^2
    N = a1 * strain + a2 * strain**2
    D = 1.0 + b1 * strain + b2 * strain**2
    
    # Safeguard against zero or negative denominator (which could cause instability)
    # If D is close to 0, the model blows up. We clip D to a small positive value.
    # However, clipping might introduce non-differentiability. 
    # Given BFGS starts at 1s, D = 1 + strain + strain^2 > 0 for strain >= 0.
    # Strain range is [-0.0019, 0.96]. 
    # If strain is negative, D could be small. 
    # Let's use a safe division.
    D_safe = np.where(np.abs(D) < 1e-6, 1e-6, D)
    
    rational_part = N / D_safe
    
    stress = rational_part + offset
    
    return stress
