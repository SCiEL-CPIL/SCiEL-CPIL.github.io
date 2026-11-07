import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed basis expansion for nonlinear oscillator dynamics.
    
    Uses a carefully chosen set of basis functions that encode known oscillator
    physics (Duffing restoring, Van der Pol damping) while maintaining the
    linear-in-parameters structure for efficient BFGS fitting.
    
    Basis functions (10 parameters):
    p0: Constant offset
    p1: Linear restoring force (x)
    p2: Cubic restoring force (Duffing) (x^3)
    p3: Linear damping (v)
    p4: Amplitude-dependent damping (x^2 * v) - Van der Pol structure
    p5: Quadratic velocity correction (v^2)
    p6: Cross term (x * v)
    p7: Smooth saturation on displacement (tanh(k1 * x))
    p8: Smooth saturation on velocity (tanh(k2 * v))
    p9: Smooth interaction (tanh(k3 * x) * v)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalize to improve conditioning
    x_center = -0.02265
    x_scale = 0.55015
    v_center = 0.05335
    v_scale = 0.22875
    
    xn = (x - x_center) / x_scale
    vn = (v - v_center) / v_scale
    
    # Basis functions
    b0 = np.ones_like(xn)
    b1 = xn                          # Linear restoring
    b2 = xn ** 3                     # Duffing cubic
    b3 = vn                          # Linear damping
    b4 = xn ** 2 * vn               # Van der Pol amplitude-dependent damping
    b5 = vn ** 2                     # Quadratic velocity
    b6 = xn * vn                     # Cross term
    b7 = np.tanh(3.0 * xn)          # Smooth displacement saturation
    b8 = np.tanh(3.0 * vn)          # Smooth velocity saturation
    b9 = np.tanh(2.0 * xn) * vn     # Smooth interaction
    
    result = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 +
              p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
    
    return result
