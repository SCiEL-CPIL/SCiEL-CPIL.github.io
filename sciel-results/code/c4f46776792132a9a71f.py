"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Hybrid basis: Polynomial, Trigonometric, and Localized functions.
    # This structural change allows the model to capture:
    # 1. Linear/Quadratic trends (poly)
    # 2. Periodic behavior (trig)
    # 3. Localized bumps or saturations (exp/tanh)
    x = np.asarray(x, dtype=float)
    
    # Ensure numerical stability
    # Basis components:
    # 0: 1 (bias)
    # 1: x
    # 2: x^2
    # 3: sin(x)
    # 4: cos(x)
    # 5: sin(2x)
    # 6: cos(2x)
    # 7: tanh(x)
    # 8: exp(-x^2)  (Gaussian centered at 0)
    # 9: exp(-(x-1.5)^2) (Gaussian centered at 1.5 to catch asymmetry)
    
    basis = np.column_stack([
        np.ones_like(x),
        x,
        x**2,
        np.sin(x),
        np.cos(x),
        np.sin(2*x),
        np.cos(2*x),
        np.tanh(x),
        np.exp(-x**2),
        np.exp(-(x - 1.5)**2)
    ])
    
    return basis @ params
# EVOLVE-BLOCK-END
