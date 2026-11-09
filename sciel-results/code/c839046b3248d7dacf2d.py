import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use Legendre polynomial basis for better numerical conditioning.
    # The basis spans the same space as the original monomial basis (total degree <= 3)
    # but is orthogonal on [-1, 1], which improves BFGS convergence and stability.
    
    # Legendre Polynomials
    # P0(t) = 1
    # P1(t) = t
    # P2(t) = 0.5 * (3*t^2 - 1)
    # P3(t) = 0.5 * (5*t^3 - 3*t)
    
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    
    P1_x = x
    P1_y = y
    
    P2_x = 0.5 * (3.0 * x2 - 1.0)
    P2_y = 0.5 * (3.0 * y2 - 1.0)
    
    P3_x = 0.5 * (5.0 * x3 - 3.0 * x)
    P3_y = 0.5 * (5.0 * y3 - 3.0 * y)
    
    # Basis terms
    # 0: 1
    # 1: P1(x)
    # 2: P1(y)
    # 3: P1(x)P1(y)
    # 4: P2(x)
    # 5: P2(y)
    # 6: P2(x)P1(y)
    # 7: P1(x)P2(y)
    # 8: P3(x)
    # 9: P3(y)
    
    pred = (params[0] * np.ones_like(x) + 
            params[1] * P1_x + 
            params[2] * P1_y + 
            params[3] * (P1_x * P1_y) + 
            params[4] * P2_x + 
            params[5] * P2_y + 
            params[6] * (P2_x * P1_y) + 
            params[7] * (P1_x * P2_y) + 
            params[8] * P3_x + 
            params[9] * P3_y)
            
    return pred
