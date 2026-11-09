"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a spectral (sine/cosine) basis.
    This representation is well-conditioned for BFGS optimization because
    sin/cos have bounded derivatives and naturally represent oscillatory dynamics.
    
    The basis includes:
    - Constant term
    - Linear terms in x and v (scaled)
    - Low-frequency sinusoidal terms in x and v
    - Cross terms combining x and v sinusoids
    - Higher-order harmonics for nonlinear effects
    """
    # Scale inputs to reasonable frequency ranges
    # x is in [-0.5728, 0.5275], v is in [-0.1754, 0.2821]
    # Scale to approximately [-pi, pi] range for sinusoidal basis
    X = x * 5.0  # Roughly maps to [-2.9, 2.6]
    V = v * 10.0  # Roughly maps to [-1.75, 2.82]
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Sinusoidal basis functions
    # Position-dependent terms
    sin_x1 = np.sin(X)
    cos_x1 = np.cos(X)
    sin_x2 = np.sin(2.0 * X)
    cos_x2 = np.cos(2.0 * X)
    
    # Velocity-dependent terms
    sin_v1 = np.sin(V)
    cos_v1 = np.cos(V)
    sin_v2 = np.sin(2.0 * V)
    cos_v2 = np.cos(2.0 * V)
    
    # Cross terms
    sin_x_v = np.sin(X + V)
    cos_x_v = np.cos(X + V)
    
    # Linear terms (useful for capturing linear dynamics)
    x_lin = X
    v_lin = V
    
    # Construct the 10-term basis:
    # 0: Constant
    # 1: sin(x) - odd nonlinearity in position
    # 2: v - linear damping/drive
    # 3: sin(v) - odd nonlinearity in velocity
    # 4: x - linear position dependence
    # 5: sin(x+v) - cross-coupling
    # 6: cos(x) - even nonlinearity in position
    # 7: sin(2x) - higher harmonic in position
    # 8: cos(v) - even nonlinearity in velocity
    # 9: sin(2v) - higher harmonic in velocity
    
    return (p0 * 1.0 +
            p1 * sin_x1 +
            p2 * v_lin +
            p3 * sin_v1 +
            p4 * x_lin +
            p5 * sin_x_v +
            p6 * cos_x1 +
            p7 * sin_x2 +
            p8 * cos_v1 +
            p9 * sin_v2)
# EVOLVE-BLOCK-END
