import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize x to [-1, 1]
    u = x / 3.0
    
    # Extract parameters
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    Ag = params[4]
    Wg = params[5]
    Cg = params[6]
    As = params[7]
    Fs = params[8]
    Ps = params[9]
    
    # Polynomial part
    poly = c0 + c1 * u + c2 * u**2 + c3 * u**3
    
    # Gaussian part: exp(-(Wg)^2 * (u - Cg)^2)
    # Using Wg**2 ensures the exponent is non-positive (decaying) if Wg is real,
    # preventing overflow for negative Wg values during optimization.
    gauss_arg = -(Wg ** 2) * (u - Cg) ** 2
    # Clip the exponent to prevent underflow/overflow issues in exp, though exp(-large) is just 0.
    # exp(large positive) is dangerous, but Wg**2 is positive, so arg is <= 0.
    # So exp(arg) is in (0, 1]. Safe.
    gauss = Ag * np.exp(gauss_arg)
    
    # Sine part
    sine = As * np.sin(Fs * u + Ps)
    
    return poly + gauss + sine
