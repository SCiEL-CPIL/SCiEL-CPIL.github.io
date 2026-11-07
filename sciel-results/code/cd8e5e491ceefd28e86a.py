# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed scaling constants to map inputs to roughly [-1, 1]
_X_SCALE = 1.0 / 0.5728
_V_SCALE = 1.0 / 0.2821

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Uses a Chebyshev polynomial basis in scaled inputs for better conditioning
    and approximation properties on bounded domains.
    """
    # Map inputs to [-1, 1]
    u = x * _X_SCALE
    w = v * _V_SCALE
    
    # Clip to [-1, 1] to ensure Chebyshev stability
    u = np.clip(u, -1.0, 1.0)
    w = np.clip(w, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_n(t) = cos(n * arccos(t))
    # Using recurrence: T_0 = 1, T_1 = t, T_{n+1} = 2t*T_n - T_{n-1}
    
    # Chebyshev for u (position)
    Tu0 = np.ones_like(u)
    Tu1 = u
    Tu2 = 2.0 * u * Tu1 - Tu0
    Tu3 = 2.0 * u * Tu2 - Tu1
    
    # Chebyshev for w (velocity)
    Tw0 = np.ones_like(w)
    Tw1 = w
    Tw2 = 2.0 * w * Tw1 - Tw0
    Tw3 = 2.0 * w * Tw2 - Tw1
    
    # Build basis: mix of Chebyshev terms in u, w, and products
    # 0: T_0(u) = 1 (bias)
    # 1: T_1(u) = u (linear restoring)
    # 2: T_2(u) = 2u^2 - 1 (quadratic restoring)
    # 3: T_3(u) = 4u^3 - 3u (cubic/Duffing restoring)
    # 4: T_1(w) = w (linear damping)
    # 5: T_2(w) = 2w^2 - 1 (quadratic damping)
    # 6: T_1(u) * T_1(w) = u*w (parametric coupling)
    # 7: T_2(u) * T_1(w) (nonlinear parametric)
    # 8: T_1(u) * T_2(w) (velocity-squared restoring)
    # 9: T_2(u) * T_2(w) (higher-order coupling)
    
    b0 = Tu0
    b1 = Tu1
    b2 = Tu2
    b3 = Tu3
    b4 = Tw1
    b5 = Tw2
    b6 = Tu1 * Tw1
    b7 = Tu2 * Tw1
    b8 = Tu1 * Tw2
    b9 = Tu2 * Tw2
    
    basis = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    return basis @ params
# EVOLVE-BLOCK-END
