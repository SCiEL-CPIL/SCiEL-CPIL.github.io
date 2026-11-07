import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    2D Chebyshev polynomial basis for nonlinear oscillator.
    
    Chebyshev polynomials are orthogonal on [-1, 1], providing better
    numerical conditioning for the BFGS optimizer compared to monomials.
    This basis allows for flexible approximation of smooth nonlinear functions.
    
    Basis functions (10 total) - 2D Chebyshev on scaled inputs:
    1. T0(x)*T0(v) = 1
    2. T1(x)*T0(v) = x_s
    3. T0(x)*T1(v) = v_s
    4. T2(x)*T0(v) = 2*x_s^2 - 1
    5. T1(x)*T1(v) = x_s*v_s
    6. T0(x)*T2(v) = 2*v_s^2 - 1
    7. T3(x)*T0(v) = 4*x_s^3 - 3*x_s
    8. T2(x)*T1(v) = (2*x_s^2-1)*v_s
    9. T1(x)*T2(v) = x_s*(2*v_s^2-1)
    10. T2(x)*T2(v) = (2*x_s^2-1)*(2*v_s^2-1)
    """
    # Scale inputs to [-1, 1] range for Chebyshev polynomials
    # x range approx [-0.57, 0.53], v range approx [-0.17, 0.28]
    x_c = 0.0
    x_r = 0.57
    v_c = 0.05
    v_r = 0.25
    
    x_s = (x - x_c) / x_r
    v_s = (v - v_c) / v_r
    
    # Clip to [-1, 1] for Chebyshev stability
    x_s = np.clip(x_s, -1.0, 1.0)
    v_s = np.clip(v_s, -1.0, 1.0)
    
    # Compute 1D Chebyshev polynomials up to degree 3
    # T0(t) = 1, T1(t) = t, T2(t) = 2t^2 - 1, T3(t) = 4t^3 - 3t
    T0x = np.ones_like(x)
    T1x = x_s
    T2x = 2.0 * x_s * x_s - 1.0
    T3x = 4.0 * x_s * x_s * x_s - 3.0 * x_s
    
    T0v = np.ones_like(v)
    T1v = v_s
    T2v = 2.0 * v_s * v_s - 1.0
    T3v = 4.0 * v_s * v_s * v_s - 3.0 * v_s
    
    # 2D Chebyshev basis terms
    b0 = T0x * T0v
    b1 = T1x * T0v
    b2 = T0x * T1v
    b3 = T2x * T0v
    b4 = T1x * T1v
    b5 = T0x * T2v
    b6 = T3x * T0v
    b7 = T2x * T1v
    b8 = T1x * T2v
    b9 = T2x * T2v
    
    # Linear combination
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * b4 + 
         params[5] * b5 + 
         params[6] * b6 + 
         params[7] * b7 + 
         params[8] * b8 + 
         params[9] * b9)
         
    return y
