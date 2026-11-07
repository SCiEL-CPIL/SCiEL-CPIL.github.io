# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a hybrid orthogonal basis.
    
    The basis consists of low-degree Chebyshev polynomials (scaled to [-1, 1])
    and low-frequency trigonometric functions orthogonal on [-2, 2].
    This provides a well-conditioned design matrix for BFGS optimization,
    combining smooth polynomial approximation with oscillatory capabilities.
    """
    # 1. Define raw basis functions
    
    # Chebyshev polynomials of the first kind T_n(t) where t = x/2
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    
    t = x / 2.0
    f0 = np.ones_like(x)
    f1 = t
    f2 = 2.0 * t**2 - 1.0
    f3 = 4.0 * t**3 - 3.0 * t
    
    # Trigonometric functions orthogonal on [-2, 2]
    # sin(pi*x/2), cos(pi*x/2) -> 1 cycle
    # sin(pi*x), cos(pi*x) -> 2 cycles
    # sin(3*pi*x/2) -> 1.5 cycles
    
    f4 = np.sin(0.5 * np.pi * x)
    f5 = np.cos(0.5 * np.pi * x)
    f6 = np.sin(np.pi * x)
    f7 = np.cos(np.pi * x)
    f8 = np.sin(1.5 * np.pi * x)
    f9 = np.cos(1.5 * np.pi * x)
    
    basis = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
    
    # 2. Compute prediction
    # The basis functions are bounded and well-conditioned.
    # We do not normalize dynamically to keep the mapping linear and stable
    # for the BFGS optimizer, relying on the orthogonality/boundedness
    # of the chosen basis.
    
    return basis @ params
# EVOLVE-BLOCK-END
