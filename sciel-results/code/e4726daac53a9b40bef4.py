import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Scale input to [-1, 1] for better conditioning of polynomial basis
    u = x / 2.0
    
    # Chebyshev polynomials of the first kind
    # T_0(u) = 1
    # T_1(u) = u
    # T_2(u) = 2u^2 - 1
    # T_3(u) = 4u^3 - 3u
    # T_4(u) = 8u^4 - 8u^2 + 1
    # T_5(u) = 16u^5 - 20u^3 + 5u
    # T_6(u) = 32u^6 - 48u^4 + 18u^2 - 1
    
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    u5 = u4 * u
    u6 = u3 * u3
    
    T0 = 1.0
    T1 = u
    T2 = 2.0 * u2 - 1.0
    T3 = 4.0 * u3 - 3.0 * u
    T4 = 8.0 * u4 - 8.0 * u2 + 1.0
    T5 = 16.0 * u5 - 20.0 * u3 + 5.0 * u
    T6 = 32.0 * u6 - 48.0 * u4 + 18.0 * u2 - 1.0
    
    # Sigmoid term
    # Center p[5], Steepness p[6], Amplitude p[4]
    # Argument: p[6] * (u - p[5])
    z = p[6] * (u - p[5])
    sigmoid_term = 0.5 * (1.0 + np.tanh(0.5 * z))
    
    # Combine
    y_hat = (p[0] * T0 + 
             p[1] * T1 + 
             p[2] * T2 + 
             p[3] * T3 +
             p[4] * sigmoid_term +
             p[7] * T4 +
             p[8] * T5 +
             p[9] * T6)
             
    return y_hat
