"""Symbolic regression with 2 inputs: Hybrid Chebyshev + Radial Basis Function basis."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 145.0


def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return a (n, 10) feature matrix combining low-order Chebyshev terms and localized RBF terms.
    
    The basis consists of:
    1. T0(x)T0(y) = 1
    2. T1(x)T0(y) = x
    3. T0(x)T1(y) = y
    4. T1(x)T1(y) = xy
    5. T2(x)T0(y) = 2x^2 - 1
    6. T0(x)T2(y) = 2y^2 - 1
    7. T2(x)T2(y) = (2x^2 - 1)(2y^2 - 1)
    
    Plus 3 localized Gaussian Radial Basis Functions (RBFs) centered at:
    8. exp(-sigma * ((x - 0.5)^2 + (y - 0.5)^2))
    9. exp(-sigma * ((x + 0.5)^2 + (y - 0.5)^2))
    10. exp(-sigma * ((x - 0.5)^2 + (y + 0.5)^2))
    
    The Chebyshev terms capture global smooth trends, while the RBF terms allow
    localized adjustments to fit non-smooth or high-frequency features that
    global polynomials struggle with. This hybrid approach aims to improve
    robustness and fit quality compared to pure polynomial bases.
    """
    n = x.shape[0]

    # Compute Chebyshev polynomials for x and y
    x2 = x * x
    y2 = y * y
    
    Tx0 = np.ones(n)
    Tx1 = x
    Tx2 = 2.0 * x2 - 1.0

    Ty0 = np.ones(n)
    Ty1 = y
    Ty2 = 2.0 * y2 - 1.0

    # Global Chebyshev features
    f0 = Tx0 * Ty0  # 1
    f1 = Tx1 * Ty0  # x
    f2 = Tx0 * Ty1  # y
    f3 = Tx1 * Ty1  # xy
    f4 = Tx2 * Ty0  # 2x^2 - 1
    f5 = Tx0 * Ty2  # 2y^2 - 1
    f6 = Tx2 * Ty2  # (2x^2 - 1)(2y^2 - 1)

    # Localized RBF features
    # Choose a bandwidth that is narrow enough to be local but not too narrow
    # to cause extreme ill-conditioning or numerical instability.
    # sigma = 10.0 gives a width of ~0.3, which is reasonable for [-1, 1].
    sigma = 10.0
    
    # Centers chosen to cover different regions of the domain
    cx1, cy1 = 0.5, 0.5
    cx2, cy2 = -0.5, 0.5
    cx3, cy3 = 0.5, -0.5

    dx1 = x - cx1
    dy1 = y - cy1
    f7 = np.exp(-sigma * (dx1 * dx1 + dy1 * dy1))

    dx2 = x - cx2
    dy2 = y - cy2
    f8 = np.exp(-sigma * (dx2 * dx2 + dy2 * dy2))

    dx3 = x - cx3
    dy3 = y - cy3
    f9 = np.exp(-sigma * (dx3 * dx3 + dy3 * dy3))

    F = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    return F


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Guard against excessive runtime
    if time.time() > _DEADLINE:
        F = _features(x, y)
        return F @ params

    F = _features(x, y)
    pred = F @ params

    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        mask = np.isfinite(pred)
        pred[~mask] = 0.0

    return pred
# EVOLVE-BLOCK-END
