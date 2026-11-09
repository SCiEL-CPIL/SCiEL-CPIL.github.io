import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod.
    
    Structure:
    1. Linear elastic term with temperature dependence.
    2. Non-linear hardening term using tanh to model yield transition and hardening.
    3. Quadratic correction term for curvature.
    4. Intercept and interaction terms.
    
    Parameters:
    p0: Elastic modulus scale
    p1: Temperature coefficient for modulus
    p2: Hardening amplitude
    p3: Temperature coefficient for hardening
    p4: Yield sharpness (inverse width of tanh transition)
    p5: Yield strain location
    p6: Quadratic strain coefficient
    p7: Temperature coefficient for quadratic term
    p8: Intercept
    p9: Strain-Temperature interaction coefficient
    """
    eps = strain
    T = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # 1. Linear Elastic Term: E(T) * eps
    # E(T) = p0 * (1 - p1 * T)
    elastic_term = p0 * (1.0 - p1 * T) * eps
    
    # 2. Plastic Hardening Term: Amplitude(T) * tanh(Sharpness * (eps - YieldStrain))
    # Amplitude(T) = p2 * (1 - p3 * T)
    # This term is ~0 for eps < p5 and grows to p2 for eps >> p5.
    # It captures the transition from elastic to plastic and subsequent hardening.
    hardening_term = p2 * (1.0 - p3 * T) * np.tanh(p4 * (eps - p5))
    
    # 3. Quadratic Correction: p6 * eps^2 * (1 - p7 * T)
    # Helps fit the curvature of the hardening region if tanh is not sufficient.
    quadratic_term = p6 * eps * eps * (1.0 - p7 * T)
    
    # 4. Interaction and Intercept
    interaction_term = p9 * eps * T
    intercept = p8
    
    return elastic_term + hardening_term + quadratic_term + interaction_term + intercept
