import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Separated Rational Stiffness
    # Linear part: p[0] * x
    # Nonlinear part: p[1] * x^3 / (1 + p[2] * x^2)
    stiff_nonlin = (p[1] * x3) / (1.0 + p[2] * x2)
    stiff = p[0] * x + stiff_nonlin
    
    # Separated Rational Damping
    # Linear part: p[3] * v
    # Nonlinear part: p[4] * v^3 / (1 + p[5] * v^2)
    damp_nonlin = (p[4] * v3) / (1.0 + p[5] * v2)
    damp = p[3] * v + damp_nonlin
    
    # Van der Pol type
    vdp = p[6] * x2 * v
    
    # Cross term
    cross = p[7] * x * v
    
    # Velocity-stiffness coupling
    v_coupling = p[8] * x * v2
    
    # Offset
    offset = p[9]
    
    return stiff + damp + vdp + cross + v_coupling + offset
