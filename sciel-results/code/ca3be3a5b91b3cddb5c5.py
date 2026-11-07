import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params: [c_v, c_x, c_x3, c_x2, c_t, c_xv, c_cos, c_sin, omega, c_bias]
    c_v = params[0]
    c_x = params[1]
    c_x3 = params[2]
    c_x2 = params[3]
    c_t = params[4]
    c_xv = params[5]
    c_cos = params[6]
    c_sin = params[7]
    omega = params[8]
    c_bias = params[9]
    
    # Calculate terms
    term_v = c_v * v
    term_x = c_x * x
    term_x3 = c_x3 * (x ** 3)
    term_x2 = c_x2 * (x ** 2)
    term_t = c_t * t
    term_xv = c_xv * (x * v)
    
    # Driving force
    phase = omega * t
    drive = c_cos * np.cos(phase) + c_sin * np.sin(phase)
    
    result = term_v + term_x + term_x3 + term_x2 + term_t + term_xv + drive + c_bias
    
    return result
