import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress model with multiplicative tanh-based hardening.
    
    sigma = E(T) * eps * (1 + A(T)*tanh(B(T)*eps) + C(T)*eps + D(T)*eps^2)
    
    This uses a multiplicative structure where the elastic response is modulated
    by a hardening factor that includes both linear and nonlinear strain terms.
    """
    eps = strain
    T = temp
    
    # Parameter unpacking
    E0 = params[0]
    c_E = params[1]
    A0 = params[2]
    c_A = params[3]
    B0 = params[4]
    c_B = params[5]
    C0 = params[6]
    c_C = params[7]
    D0 = params[8]
    c_D = params[9]
    
    # Temperature-dependent coefficients
    E_T = E0 * (1.0 + c_E * T)
    A_T = A0 * (1.0 + c_A * T)
    B_T = B0 * (1.0 + c_B * T)
    C_T = C0 * (1.0 + c_C * T)
    D_T = D0 * (1.0 + c_D * T)
    
    # Clip B_T * eps to prevent tanh saturation issues
    arg = B_T * eps
    arg = np.clip(arg, -20, 20)
    
    # Hardening factor
    hardening = 1.0 + A_T * np.tanh(arg) + C_T * eps + D_T * (eps * eps)
    
    stress = E_T * eps * hardening
    
    return stress
