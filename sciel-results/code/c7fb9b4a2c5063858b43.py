import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model using multiplicative temperature-modulated elastic-plastic
    formulation with sinh-based hardening.
    
    Structural change: Replace additive polynomial basis with a physically-motivated
    multiplicative model:
    - Temperature-dependent elastic modulus E(T) = p0 + p1*T + p2*T^2
    - Temperature-dependent yield stress sigma_y(T) = p3 + p4*T + p5*T^2
    - Smooth hardening via sinh(p6*(eps - eps_y)) with saturation
    - Temperature modulates the hardening rate and saturation level
    
    Parameters:
    p0: Elastic modulus at reference temperature
    p1: Linear temperature coefficient for modulus
    p2: Quadratic temperature coefficient for modulus
    p3: Yield stress at reference temperature
    p4: Linear temperature coefficient for yield stress
    p5: Quadratic temperature coefficient for yield stress
    p6: Hardening rate (controls sigmoid-like transition sharpness)
    p7: Yield strain location
    p8: Hardening saturation scale
    p9: Intercept
    """
    eps = strain
    T = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent elastic modulus
    E_T = p0 + p1 * T + p2 * T * T
    
    # Temperature-dependent yield stress
    sigma_y_T = p3 + p4 * T + p5 * T * T
    
    # Plastic strain beyond yield
    eps_plastic = np.maximum(eps - p7, 0.0)
    
    # Smooth hardening using sinh with saturation via tanh
    # sinh gives smooth transition, tanh provides saturation
    hardening_arg = p6 * eps_plastic
    # Clip to avoid overflow in sinh
    hardening_arg = np.clip(hardening_arg, -50, 50)
    hardening = np.sinh(hardening_arg)
    
    # Saturation: scale hardening so it doesn't grow unbounded
    # Use 1 - exp(-p8 * eps_plastic) as saturation factor
    sat_arg = np.clip(p8 * eps_plastic, -50, 50)
    saturation = 1.0 - np.exp(-sat_arg)
    
    # Temperature modulates hardening amplitude
    hardening_amp = sigma_y_T * (1.0 + 0.1 * (T - 0.5))
    
    # Elastic response
    elastic = E_T * eps
    
    # Plastic hardening contribution (only for eps > yield strain)
    # The hardening adds to stress beyond the elastic prediction at yield
    plastic = hardening_amp * hardening * saturation
    
    # Total stress
    result = p9 + elastic + plastic
    
    return result
