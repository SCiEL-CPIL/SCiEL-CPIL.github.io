import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod using a rational function.
    
    This model uses a ratio of polynomials to capture the non-linear
    stress-strain behavior, including the elastic-plastic transition.
    
    sigma = N(strain, temp) / D(strain, temp)
    
    N = strain * [ (p0 + p1*T) + (p2 + p3*T)*strain + (p4 + p5*T)*strain^2 ]
    D = 1 + (p6 + p7*T)*strain + (p8 + p9*T)*strain^2
    
    Parameters (10):
    p0, p1: Coefficients for the linear strain term in numerator (Elastic slope approx)
    p2, p3: Coefficients for the quadratic strain term in numerator
    p4, p5: Coefficients for the cubic strain term in numerator
    p6, p7: Coefficients for the linear strain term in denominator (Softening/Transition)
    p8, p9: Coefficients for the quadratic strain term in denominator
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale parameters to ensure good gradient magnitudes for BFGS starting from ones
    # The inputs are in [0, 1], so unscaled params near 1 might result in small outputs
    # or flat gradients. Scaling by 10 helps the optimizer see the effect of parameters.
    scale = 10.0
    p0 *= scale
    p1 *= scale
    p2 *= scale
    p3 *= scale
    p4 *= scale
    p5 *= scale
    p6 *= scale
    p7 *= scale
    p8 *= scale
    p9 *= scale
    
    u = strain
    T = temp
    
    # Numerator: Polynomial in u with T-dependent coefficients
    # We factor out u to ensure sigma(0) = 0
    a0 = p0 + p1 * T
    a1 = p2 + p3 * T
    a2 = p4 + p5 * T
    
    num = u * (a0 + a1 * u + a2 * u**2)
    
    # Denominator: Polynomial in u with T-dependent coefficients
    # Constant term is 1 to ensure finite slope at u=0
    b0 = 1.0
    b1 = p6 + p7 * T
    b2 = p8 + p9 * T
    
    den = b0 + b1 * u + b2 * u**2
    
    # Safe division to handle potential zeros in denominator (though unlikely with b0=1)
    # We add a small epsilon to prevent division by zero if the optimizer explores bad regions
    # Note: np.where is safer for array operations
    sigma = num / (den + 1e-12)
    
    return sigma
