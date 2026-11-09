import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a combined polynomial-tanh basis
    with input scaling for numerical conditioning.
    
    This approach blends:
    - Polynomial terms (cubic) for global Duffing-like stiffness and nonlinear damping
    - tanh-based terms for smooth saturation behavior
    - Cross-coupling terms for interaction between position and velocity
    """
    # Scale inputs to roughly [-1, 1] range to help BFGS conditioning
    # x range approx [-0.57, 0.53], v range approx [-0.18, 0.28]
    X = x / 0.6
    V = v / 0.3
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial basis terms (cubic)
    poly_0 = 1.0
    poly_1 = X
    poly_2 = V
    poly_3 = X**2
    poly_4 = X * V
    poly_5 = V**2
    poly_6 = X**3
    poly_7 = X**2 * V
    poly_8 = X * V**2
    poly_9 = V**3
    
    # tanh-based saturation terms
    tanh_X = np.tanh(X)
    tanh_V = np.tanh(V)
    X_tanh_X = X * np.tanh(X**2)
    V_tanh_V = V * np.tanh(V**2)
    
    # Combined basis: mix of polynomial and tanh terms
    # p0: bias
    # p1: linear restoring (poly)
    # p2: linear damping (poly)
    # p3: quadratic restoring (poly)
    # p4: cross-coupling linear (poly)
    # p5: quadratic damping (poly)
    # p6: cubic restoring (Duffing, poly)
    # p7: saturated restoring (tanh-based)
    # p8: saturated damping (tanh-based)
    # p9: cross-coupling saturated
    
    term_0 = p0 * poly_0
    term_1 = p1 * poly_1
    term_2 = p2 * poly_2
    term_3 = p3 * poly_3
    term_4 = p4 * poly_4
    term_5 = p5 * poly_5
    term_6 = p6 * poly_6
    term_7 = p7 * (X_tanh_X + 0.5 * tanh_X)
    term_8 = p8 * (V_tanh_V + 0.5 * tanh_V)
    term_9 = p9 * (np.tanh(X) * np.tanh(V))
    
    return term_0 + term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8 + term_9
