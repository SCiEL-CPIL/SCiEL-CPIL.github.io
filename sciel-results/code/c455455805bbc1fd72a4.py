"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a hybrid polynomial-trigonometric model.
    
    Uses scaled inputs and combines polynomial terms with trigonometric terms
    to capture both the algebraic nonlinearity and periodic nature of oscillators.
    """
    # Scale inputs based on training ranges for numerical conditioning
    # x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    X = x / 0.55
    V = v / 0.25
    
    # Polynomial terms
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    XV = X * V
    
    # Trigonometric terms (natural for oscillatory systems)
    sinX = np.sin(X)
    cosX = np.cos(X)
    sinV = np.sin(V)
    
    # Hybrid model: combine polynomial and trigonometric features
    # p0: constant offset
    # p1: linear in X
    # p2: cubic in X (Duffing-like)
    # p3: linear in V (damping)
    # p4: V*X (coupling)
    # p5: sin(X) (periodic restoring)
    # p6: cos(X) - 1 (symmetric periodic)
    # p7: V^2 (quadratic damping)
    # p8: X^2 (quadratic restoring)
    # p9: sin(V) (velocity periodicity)
    
    result = (
        params[0]
        + params[1] * X
        + params[2] * X3
        + params[3] * V
        + params[4] * XV
        + params[5] * sinX
        + params[6] * (cosX - 1.0)
        + params[7] * V2
        + params[8] * X2
        + params[9] * sinV
    )
    
    return result
# EVOLVE-BLOCK-END
