import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model with time-dependent driving force.
    
    Model:
    a = p1 + p2*x + p3*v + p4*x^3 + p5*v^3 + p6*x^5 + p7*v^5 + p8*sin(p0*t) + p9*cos(p0*t)
    
    Parameters:
    p0: Frequency of the driving force
    p1: Constant offset
    p2: Linear restoring force coefficient
    p3: Linear damping coefficient
    p4: Cubic restoring force coefficient (Duffing term)
    p5: Cubic damping coefficient
    p6: Quintic restoring force coefficient
    p7: Quintic damping coefficient
    p8: Amplitude of sine driving component
    p9: Amplitude of cosine driving component
    """
    w = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Pre-compute powers for efficiency
    x2 = x * x
    x3 = x2 * x
    
    v2 = v * v
    v3 = v2 * v
    
    # Driving force components
    wt = w * t
    sin_wt = np.sin(wt)
    cos_wt = np.cos(wt)
    
    # Construct the equation
    # Using up to quadratic and cubic terms to improve conditioning
    # and capture broader non-linearities without high-order instability.
    
    result = (
        p1 + 
        p2 * x + 
        p3 * v + 
        p4 * x2 + 
        p5 * v2 + 
        p6 * x3 + 
        p7 * v3 + 
        p8 * sin_wt + 
        p9 * cos_wt
    )
    
    return result
