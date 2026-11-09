import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use Chebyshev polynomial basis for better conditioning
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    
    Tx0 = 1.0
    Tx1 = x
    Tx2 = 2.0 * x * x - 1.0
    Tx3 = 4.0 * x * x * x - 3.0 * x
    
    Ty0 = 1.0
    Ty1 = y
    Ty2 = 2.0 * y * y - 1.0
    Ty3 = 4.0 * y * y * y - 3.0 * y
    
    p = params
    
    return (p[0] * Tx0 * Ty0 + 
            p[1] * Tx1 * Ty0 + 
            p[2] * Tx0 * Ty1 + 
            p[3] * Tx2 * Ty0 + 
            p[4] * Tx1 * Ty1 + 
            p[5] * Tx0 * Ty2 + 
            p[6] * Tx3 * Ty0 + 
            p[7] * Tx2 * Ty1 + 
            p[8] * Tx1 * Ty2 + 
            p[9] * Tx0 * Ty3)
