import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physically-motivated stress-strain model for aluminium.
    
    Model: sigma = E(t)*s + sigma_y(t) * (1 - exp(-n * eps_p / eps_0))
    where eps_p = max(0, s - s_y) is the plastic strain.
    
    Parameters (10 total):
    params[0]  = E0          (base elastic modulus, ~1.0)
    params[1]  = E1          (temperature coefficient of modulus)
    params[2]  = sy0         (base yield stress)
    params[3]  = sy1         (temperature coefficient of yield stress)
    params[4]  = n0          (base hardening exponent)
    params[5]  = n1          (temperature coefficient of hardening exponent)
    params[6]  = eps0        (characteristic plastic strain scale)
    params[7]  = s_y         (yield strain)
    params[8]  = c1          (quadratic elastic curvature coefficient)
    params[9]  = c2          (cubic elastic curvature coefficient)
    """
    s = strain
    t = temp
    
    # Extract parameters
    E0 = params[0]
    E1 = params[1]
    sy0 = params[2]
    sy1 = params[3]
    n0 = params[4]
    n1 = params[5]
    eps0 = params[6]
    s_y = params[7]
    c1 = params[8]
    c2 = params[9]
    
    # Ensure positive scales
    eps0 = np.maximum(eps0, 1e-3)
    
    # Temperature-dependent parameters
    E = E0 + E1 * t
    sy = sy0 + sy1 * t
    n = n0 + n1 * t
    
    # Plastic strain
    eps_p = np.maximum(0.0, s - s_y)
    
    # Elastic part with slight curvature
    elastic = E * s + c1 * s**2 + c2 * s**3
    
    # Plastic part: nonlinear hardening
    # Use a smooth approximation: sy * (1 - exp(-n * eps_p / eps0))
    # This gives a smooth transition from elastic to plastic
    plastic = sy * (1.0 - np.exp(-n * eps_p / eps0))
    
    # Total stress
    stress = elastic + plastic
    
    return stress
