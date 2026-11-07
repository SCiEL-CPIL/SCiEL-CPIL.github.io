"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Smooth bump-function basis for stress-strain of an aluminium rod.
    
    Structural change: instead of piecewise-linear ReLU knots (which are
    non-differentiable and may cause BFGS to struggle with gradient estimates),
    use smooth Gaussian/RBF-like bump functions centered at key strain values.
    
    This gives a fully smooth, infinitely differentiable surface that BFGS
    can optimize efficiently. The bumps capture local features (yield, hardening
    onset, plateau) while maintaining global smoothness.
    
    Basis functions:
    - Global: 1, s, t
    - Bumps in strain: exp(-(s-c1)^2/w1), exp(-(s-c2)^2/w2), exp(-(s-c3)^2/w3)
    - Bump*temp interactions: exp(-(s-c1)^2/w1)*t, exp(-(s-c2)^2/w2)*t, exp(-(s-c3)^2/w3)*t
    - Cross term: s*t
    
    Centers chosen for aluminium: c1=0.01 (yield), c2=0.1 (hardening), c3=0.5 (plateau)
    Widths: w1=0.005, w2=0.02, w3=0.1
    """
    s = strain
    t = temp
    
    # Centers and widths for bump functions
    c1, w1 = 0.01, 0.005
    c2, w2 = 0.1, 0.02
    c3, w3 = 0.5, 0.1
    
    # Bump functions (smooth Gaussians)
    b1 = np.exp(-((s - c1) ** 2) / w1)
    b2 = np.exp(-((s - c2) ** 2) / w2)
    b3 = np.exp(-((s - c3) ** 2) / w3)
    
    # Basis: 1, s, t, s*t, bump1, bump2, bump3, bump1*t, bump2*t, bump3*t
    b0 = np.ones_like(s)
    b4 = s
    b5 = t
    b6 = s * t
    b7 = b1 * t
    b8 = b2 * t
    b9 = b3 * t
    
    return (params[0] * b0 + params[1] * b4 + params[2] * b5 + params[3] * b6 +
            params[4] * b1 + params[5] * b2 + params[6] * b3 +
            params[7] * b7 + params[8] * b8 + params[9] * b9)
# EVOLVE-BLOCK-END
