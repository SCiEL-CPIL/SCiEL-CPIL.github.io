import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Physics-based basis for nonlinear oscillator acceleration
    # a(t, x, v) = Drive(t) + Damping(v) + Restoring(x) + Constant
    
    # Drive terms (assuming frequency 1, allowing phase shift)
    drive_cos = np.cos(t)
    drive_sin = np.sin(t)
    
    # Damping terms (linear, quadratic, cubic)
    damp_lin = v
    damp_quad = v**2
    damp_cub = v**3
    
    # Restoring force terms (linear, nonlinear sin, cubic correction)
    # For small x, sin(x) ~ x - x^3/6. 
    # Including both x and sin(x) allows fitting the exact curvature.
    # Including x^3 allows for Duffing-type hardening/softening.
    rest_lin = x
    rest_nonlin = np.sin(x)
    rest_cub = x**3
    
    # Constant offset
    const = 1.0
    
    # Interaction term (often 0, but allows for slight gyroscopic effects or fitting noise)
    cross = x * v
    
    # Combine with parameters
    # p0: drive_cos
    # p1: drive_sin
    # p2: damp_lin
    # p3: damp_quad
    # p4: damp_cub
    # p5: rest_lin
    # p6: rest_nonlin
    # p7: rest_cub
    # p8: const
    # p9: cross
    
    result = (params[0] * drive_cos + 
              params[1] * drive_sin + 
              params[2] * damp_lin + 
              params[3] * damp_quad + 
              params[4] * damp_cub + 
              params[5] * rest_lin + 
              params[6] * rest_nonlin + 
              params[7] * rest_cub + 
              params[8] * const + 
              params[9] * cross)
              
    return result
