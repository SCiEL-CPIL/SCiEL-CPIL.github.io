import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model using piecewise-linear (ramp) basis functions.
    
    Uses clipped linear terms to model yield point and hardening regimes
    in a differentiable manner suitable for BFGS optimization.
    
    Parameters:
    p0: elastic modulus
    p1: temperature coefficient for modulus
    p2, p3: yield strain breakpoints (temperature-dependent)
    p4, p5: hardening slopes after first yield
    p6, p7: second yield breakpoints
    p8, p9: saturation offset and temperature interaction
    """
    eps = strain
    T = temp
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent yield points
    # First yield at strain ~ p2 + p3*T
    y1 = p2 + p3 * T
    # Second yield at strain ~ p6 + p7*T
    y2 = p6 + p7 * T
    
    # Elastic region: linear up to first yield
    elastic = p0 * (1.0 - p1 * T) * np.minimum(eps, y1)
    
    # Hardening region 1: ramp from y1 to y2
    ramp1 = np.maximum(eps - y1, 0.0)
    hardening1 = p4 * ramp1
    
    # Hardening region 2: ramp from y2 onwards
    ramp2 = np.maximum(eps - y2, 0.0)
    hardening2 = p5 * ramp2
    
    # Temperature-dependent hardening modulation
    temp_mod = 1.0 + p8 * T
    
    # Combine: elastic + hardening with temperature modulation
    stress = elastic + temp_mod * (hardening1 + hardening2)
    
    # Add constant offset with temperature dependence
    offset = p9 * (1.0 - 0.5 * T)
    
    return stress + offset
