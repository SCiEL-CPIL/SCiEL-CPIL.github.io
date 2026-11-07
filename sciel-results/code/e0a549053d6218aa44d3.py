"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Fourier Series approach
    # x is in [-3, 3]. We map this to phases in [0, 2*pi] or similar.
    # Let's use base frequency w = pi / 3 so that x=3 corresponds to phase pi.
    # Actually, x=3 -> pi, x=-3 -> -pi. This covers one full period for k=1? No, sin(-pi)=0, sin(pi)=0.
    # Let's just use standard harmonics on the scaled variable u = x/3.
    # u in [-1, 1].
    # Basis: 1, sin(pi*u), cos(pi*u), sin(2*pi*u), cos(2*pi*u), ...
    
    u = x / 3.0
    
    # Precompute basis terms
    # p0: 1
    # p1: sin(pi*u)
    # p2: cos(pi*u)
    # p3: sin(2*pi*u)
    # p4: cos(2*pi*u)
    # p5: sin(3*pi*u)
    # p6: cos(3*pi*u)
    # p7: sin(4*pi*u)
    # p8: cos(4*pi*u)
    # p9: sin(5*pi*u)
    
    result = np.zeros_like(x)
    
    # Constant term
    result += params[0]
    
    # Harmonics
    # k=1
    result += params[1] * np.sin(np.pi * u)
    result += params[2] * np.cos(np.pi * u)
    
    # k=2
    result += params[3] * np.sin(2 * np.pi * u)
    result += params[4] * np.cos(2 * np.pi * u)
    
    # k=3
    result += params[5] * np.sin(3 * np.pi * u)
    result += params[6] * np.cos(3 * np.pi * u)
    
    # k=4
    result += params[7] * np.sin(4 * np.pi * u)
    result += params[8] * np.cos(4 * np.pi * u)
    
    # k=5 (sine only to fill 10 params)
    result += params[9] * np.sin(5 * np.pi * u)
    
    return result
# EVOLVE-BLOCK-END
