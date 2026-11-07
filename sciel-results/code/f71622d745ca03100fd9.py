"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a Van der Pol-like nonlinear oscillator.
    
    Strategy: Use a physics-informed polynomial basis that explicitly models
    the limit-cycle dynamics of a Van der Pol oscillator. The basis includes:
    1. Constant bias
    2. Linear restoring force (spring)
    3. Linear damping (viscous)
    4. Cubic restoring force (Duffing nonlinearity)
    5. Quadratic velocity term (energy injection/dissipation)
    6. Cross term x*v (parametric coupling)
    7. Quintic restoring force (higher order saturation)
    8. Cubic velocity term (nonlinear damping)
    9. Quadratic displacement * velocity term
    10. Quartic displacement term
    
    This structure is more globally consistent and physically motivated than
    generic local bases like RBFs, potentially allowing BFGS to converge to
    a better global minimum by respecting the underlying physics of the system.
    """
    p = params
    
    # Normalize inputs to improve conditioning
    # x range: [-0.5728, 0.5275] -> center ~0, scale ~0.55
    # v range: [-0.1754, 0.2821] -> center ~0, scale ~0.25
    x_n = x / 0.55
    v_n = v / 0.25
    
    # Compute basis functions
    # The goal is to capture the Van der Pol equation: 
    # x'' = mu * (1 - x^2) * x' - x
    # Expanded: x'' = -x + mu*x' - mu*x^2*x'
    # We use a richer set of terms to allow for more general nonlinearities.
    
    # Term 0: Bias
    t0 = 1.0
    
    # Term 1: Linear displacement (restoring force)
    t1 = x_n
    
    # Term 2: Linear velocity (damping)
    t2 = v_n
    
    # Term 3: Cubic displacement (Duffing term)
    t3 = x_n ** 3
    
    # Term 4: Quadratic velocity (nonlinear damping/injection)
    t4 = v_n ** 2
    
    # Term 5: Cross term x*v
    t5 = x_n * v_n
    
    # Term 6: Quintic displacement (saturation)
    t6 = x_n ** 5
    
    # Term 7: Cubic velocity
    t7 = v_n ** 3
    
    # Term 8: x^2 * v (Van der Pol specific structure: -mu*x^2*x')
    t8 = (x_n ** 2) * v_n
    
    # Term 9: Quartic displacement
    t9 = x_n ** 4
    
    # Combine with parameters
    # p[0] to p[9] correspond to t0 to t9
    out = (p[0] * t0 + 
           p[1] * t1 + 
           p[2] * t2 + 
           p[3] * t3 + 
           p[4] * t4 + 
           p[5] * t5 + 
           p[6] * t6 + 
           p[7] * t7 + 
           p[8] * t8 + 
           p[9] * t9)
           
    return out
# EVOLVE-BLOCK-END
