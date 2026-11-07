"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using 10 basis functions to approximate an unknown function.
    Basis: orthogonal polynomials (shifted) + trigonometric + tanh.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    # 1. Constant
    f0 = np.ones_like(x)
    # 2. Linear
    f1 = x
    # 3. Quadratic (shifted for orthogonality on [-3,3])
    f2 = x**2 - 3.0
    # 4. Cubic (shifted for orthogonality)
    f3 = x**3 - 2.7 * x
    # 5. Sine
    f4 = np.sin(x)
    # 6. Cosine
    f5 = np.cos(x)
    # 7. Sine 2x
    f6 = np.sin(2.0 * x)
    # 8. Cosine 2x
    f7 = np.cos(2.0 * x)
    # 9. Tanh
    f8 = np.tanh(x)
    # 10. Sine 3x
    f9 = np.sin(3.0 * x)

    # Build feature matrix (N, 10)
    features = np.column_stack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9))

    # Linear combination
    return features @ params

# EVOLVE-BLOCK-END
