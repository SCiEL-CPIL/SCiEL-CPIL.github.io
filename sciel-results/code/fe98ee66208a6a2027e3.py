import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for the acceleration of a nonlinear oscillator.
    
    Uses Chebyshev polynomial basis of degree 3 in scaled position and velocity,
    combined with cross-coupling terms and sinusoidal time driving.
    
    Scaling maps training ranges to [-1, 1] for numerical stability.
    """
    # Scale inputs to [-1, 1] based on training ranges
    mid_x = (0.1749 + (-0.2056)) / 2.0
    half_x = (0.1749 - (-0.2056)) / 2.0
    mid_v = (0.3624 + (-0.3391)) / 2.0
    half_v = (0.3624 - (-0.3391)) / 2.0
    mid_t = (50.0 + 30.0) / 2.0
    half_t = (50.0 - 30.0) / 2.0
    
    x_s = (x - mid_x) / half_x
    v_s = (v - mid_v) / half_v
    t_s = (t - mid_t) / half_t
    
    # Chebyshev polynomials: T1(z)=z, T2(z)=2z^2-1, T3(z)=4z^3-3z
    T1_x = x_s
    T2_x = 2.0 * x_s * x_s - 1.0
    T3_x = 4.0 * x_s * x_s * x_s - 3.0 * x_s
    
    T1_v = v_s
    T2_v = 2.0 * v_s * v_s - 1.0
    T3_v = 4.0 * v_s * v_s * v_s - 3.0 * v_s
    
    # Cross-coupling terms
    cross_xv = T1_x * T1_v
    cross_x2v = T2_x * T1_v
    cross_xv2 = T1_x * T2_v
    
    # Sinusoidal time terms
    sin_t = np.sin(t_s)
    cos_t = np.cos(t_s)
    
    return (
        params[0] * T1_x
        + params[1] * T2_x
        + params[2] * T3_x
        + params[3] * T1_v
        + params[4] * T2_v
        + params[5] * T3_v
        + params[6] * cross_xv
        + params[7] * cross_x2v
        + params[8] * cross_xv2
        + params[9] * sin_t
    )
