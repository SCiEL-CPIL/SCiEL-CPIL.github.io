import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a physics-motivated Chebyshev basis.
    Focuses on odd-order terms for restoring force and damping,
    plus interaction terms, which are typical for nonlinear oscillators.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Map x to [-1, 1]
    # x range: [-0.5728, 0.5275]
    x_center = -0.02265
    x_scale = 0.55015
    X = (x - x_center) / x_scale
    
    # Map v to [-1, 1]
    # v range: [-0.1754, 0.2821]
    v_center = 0.05335
    v_scale = 0.22875
    V = (v - v_center) / v_scale
    
    # Clip to avoid numerical issues if slightly out of range, though inputs are in range
    # X = np.clip(X, -1.0, 1.0)
    # V = np.clip(V, -1.0, 1.0)

    # Chebyshev polynomials of the first kind
    # T0 = 1
    # T1 = x
    # T2 = 2x^2 - 1
    # T3 = 4x^3 - 3x
    # T5 = 16x^5 - 20x^3 + 5x
    # T7 = 64x^7 - 112x^5 + 56x^3 - 7x
    
    # Calculate odd Chebyshev polynomials for X (Restoring force candidates)
    T1_X = X
    T3_X = 4.0 * X**3 - 3.0 * X
    T5_X = 16.0 * X**5 - 20.0 * X**3 + 5.0 * X
    
    # Calculate odd Chebyshev polynomials for V (Damping candidates)
    T1_V = V
    T3_V = 4.0 * V**3 - 3.0 * V
    T5_V = 16.0 * V**5 - 20.0 * V**3 + 5.0 * V
    
    # Coupling terms
    # Linear coupling
    C1 = X * V
    # Nonlinear coupling 1: X^3 * V
    C2 = T3_X * V
    # Nonlinear coupling 2: X * V^3
    C3 = X * T3_V
    
    # Basis:
    # 0: Constant
    # 1: T1(X)
    # 2: T3(X)
    # 3: T5(X)
    # 4: T1(V)
    # 5: T3(V)
    # 6: T5(V)
    # 7: C1 (X*V)
    # 8: C2 (T3_X*V)
    # 9: C3 (X*T3_V)
    
    return (p0 * 1.0 + 
            p1 * T1_X + 
            p2 * T3_X + 
            p3 * T5_X + 
            p4 * T1_V + 
            p5 * T3_V + 
            p6 * T5_V + 
            p7 * C1 + 
            p8 * C2 + 
            p9 * C3)
