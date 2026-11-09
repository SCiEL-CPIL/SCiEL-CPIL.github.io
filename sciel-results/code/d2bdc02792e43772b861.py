import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation of motion for a damped nonlinear oscillator with periodic driving.
    
    Parameters:
    t: Time
    x: Position
    v: Velocity
    params: Array of 10 fitted constants
        p[0]: Amplitude of 1st harmonic drive (cos)
        p[1]: Frequency of drive
        p[2]: Amplitude of 1st harmonic drive (sin)
        p[3]: Linear spring constant
        p[4]: Cubic spring constant
        p[5]: Linear damping constant
        p[6]: Cubic damping constant
        p[7]: Quadratic spring constant
        p[8]: Cross-term damping (x*v)
        p[9]: Bias (DC offset)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Drive terms (1st harmonic as linear combo of sin and cos)
    drive = p0 * np.cos(p1 * t) + p2 * np.sin(p1 * t)
    
    # Spring terms
    spring = p3 * x + p7 * x**2 + p4 * x**3
    
    # Damping terms
    damping = p5 * v + p8 * x * v + p6 * v**3
    
    # Bias
    bias = p9
    
    return drive + spring + damping + bias
