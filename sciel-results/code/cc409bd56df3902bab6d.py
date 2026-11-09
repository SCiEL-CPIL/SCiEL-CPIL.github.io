# EVOLVE-BLOCK-START
import numpy as np

# Precompute fixed RBF centers (5 centers) spread across the input domain
# x range: [-0.5728, 0.5275], v range: [-0.1754, 0.2821]
_X0, _X1 = -0.5728, 0.5275
_V0, _V1 = -0.1754, 0.2821

# 5 centers arranged in a quincunx pattern
_CX = np.array([0.0,
                (_X0 + _X1) / 2.0,
                (_X0 + _X1) / 2.0,
                _X0 + 0.25 * (_X1 - _X0),
                _X0 + 0.75 * (_X1 - _X0)])
_CV = np.array([(_V0 + _V1) / 2.0,
                _V0 + 0.25 * (_V1 - _V0),
                _V0 + 0.75 * (_V1 - _V0),
                (_V0 + _V1) / 2.0,
                (_V0 + _V1) / 2.0])

# Fixed Gaussian widths (inverse of typical local feature scale)
# Chosen so RBFs have meaningful support without being too narrow
_WX = 1.8   # 1/sigma_x
_WV = 6.0   # 1/sigma_v


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sparse RBF + global quadratic trend for nonlinear oscillator acceleration.

    params (length 10):
      p0: global constant
      p1: global linear in x
      p2: global linear in v
      p3: global quadratic x^2
      p4: global quadratic v^2
      p5..p9: RBF weights for 5 fixed centers
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Global quadratic trend
    trend = p0 + p1 * x + p2 * v + p3 * x * x + p4 * v * v

    # Sparse RBF terms (Gaussian kernels with fixed centers/widths)
    dx = x - _CX
    dv = v - _CV
    r2 = _WX * _WX * dx * dx + _WV * _WV * dv * dv

    rbf = (p5 * np.exp(-r2[:, 0]) +
           p6 * np.exp(-r2[:, 1]) +
           p7 * np.exp(-r2[:, 2]) +
           p8 * np.exp(-r2[:, 3]) +
           p9 * np.exp(-r2[:, 4]))

    return trend + rbf
# EVOLVE-BLOCK-END
