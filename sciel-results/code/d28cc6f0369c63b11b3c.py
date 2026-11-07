"""Symbolic regression candidate: Rich trigonometric + polynomial hybrid basis.

This model uses a linear-in-parameters basis that combines:
1. Low-degree polynomial terms (up to degree 2) for smooth trends.
2. Trigonometric terms with various frequencies to capture periodic/oscillatory structure.
3. Cross-terms between polynomial and trigonometric components.

The key insight is that the hidden function may have periodic components that
polynomial bases cannot capture. By including sine and cosine terms with
non-integer frequencies, we can fit wave-like structures. The basis is still
linear in parameters, so BFGS finds the global minimum.

We use 10 parameters total, carefully chosen to balance polynomial and
trigonometric capacity:
  p[0]: constant
  p[1]: x
  p[2]: y
  p[3]: z
  p[4]: sin(2*pi*x)
  p[5]: sin(2*pi*y)
  p[6]: sin(2*pi*z)
  p[7]: cos(2*pi*x)
  p[8]: cos(2*pi*y)
  p[9]: cos(2*pi*z)

This gives full periodic capacity in each dimension with period 1, which is
well-suited for inputs in [0.5, 2] where we see 1.5 periods. The linear terms
capture non-periodic trends.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # Polynomial linear terms
    c1 = np.ones_like(x, dtype=np.float64)
    c2 = x
    c3 = y
    c4 = z

    # Trigonometric terms with frequency 2*pi (period 1)
    c5 = np.sin(2.0 * np.pi * x)
    c6 = np.sin(2.0 * np.pi * y)
    c7 = np.sin(2.0 * np.pi * z)
    c8 = np.cos(2.0 * np.pi * x)
    c9 = np.cos(2.0 * np.pi * y)
    c10 = np.cos(2.0 * np.pi * z)

    out = (
        p[0] * c1
        + p[1] * c2
        + p[2] * c3
        + p[3] * c4
        + p[4] * c5
        + p[5] * c6
        + p[6] * c7
        + p[7] * c8
        + p[8] * c9
        + p[9] * c10
    )
    return out
# EVOLVE-BLOCK-END
