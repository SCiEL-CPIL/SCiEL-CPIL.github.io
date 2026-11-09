"""Symbolic regression with 1 input(s) using a hybrid basis (poly, RBF, Fourier, exp)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a linear combination of a hybrid basis:
    - Chebyshev polynomials (global smoothness)
    - Radial basis functions (local flexibility)
    - Fourier terms (oscillatory components)
    - Exponential terms (decaying/growing components)
    
    params is an array of length 10 (as per the contract).
    However, to leverage the hybrid basis effectively within the 10-parameter constraint,
    we must be strategic. The contract says "params, an array of length 10".
    
    Wait, looking at the inspirations, they used different bases.
    The current program uses 10 Chebyshev coefficients.
    
    If I change the basis, I must still use exactly 10 parameters.
    
    Let's design a 10-term hybrid basis that is likely to capture a wider range of functions:
    1. Constant: 1
    2. Linear: x
    3. Quadratic: x^2
    4. Cubic: x^3
    5. RBF centered at 1.0: exp(-((x-1.0)/0.8)^2)
    6. RBF centered at 3.0: exp(-((x-3.0)/0.8)^2)
    7. RBF centered at 5.0: exp(-((x-5.0)/0.8)^2)
    8. Sin: sin(2*pi*x / 6.0)  (one period over domain)
    9. Cos: cos(2*pi*x / 6.0)
    10. Exp: exp(-x) (decaying exponential)
    
    This basis covers:
    - Polynomials (1-4)
    - Local bumps (5-7)
    - Oscillations (8-9)
    - Exponential decay (10)
    
    This is a much richer basis than pure Chebyshev for non-polynomial targets.
    """
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    n_points = x.shape[0]
    if n_points == 0:
        return np.array([])
        
    # Build the 10-term basis matrix
    basis = np.zeros((n_points, 10), dtype=np.float64)
    
    # 1-4: Polynomials
    basis[:, 0] = 1.0
    basis[:, 1] = x
    basis[:, 2] = x**2
    basis[:, 3] = x**3
    
    # 5-7: RBFs with width 0.8, centers at 1, 3, 5
    width = 0.8
    centers = [1.0, 3.0, 5.0]
    for i, c in enumerate(centers):
        basis[:, 4+i] = np.exp(-((x - c) / width)**2)
        
    # 8-9: Fourier (one period over [0.1, 6])
    # Period T = 6.0 - 0.1 = 5.9 approx, or just use 6.0 for simplicity
    # Let's use frequency 2*pi/6.0
    freq = 2.0 * np.pi / 6.0
    basis[:, 7] = np.sin(freq * x)
    basis[:, 8] = np.cos(freq * x)
    
    # 10: Exponential decay
    basis[:, 9] = np.exp(-x)
    
    # Linear combination
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
