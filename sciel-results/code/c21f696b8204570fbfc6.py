import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Legendre polynomials of the first kind
    # P0(x) = 1
    # P1(x) = x
    # P2(x) = 0.5 * (3x^2 - 1)
    # P3(x) = 0.5 * (5x^3 - 3x)
    
    P0_x = np.ones_like(x)
    P1_x = x
    P2_x = 0.5 * (3 * x**2 - 1)
    P3_x = 0.5 * (5 * x**3 - 3 * x)
    
    P0_y = np.ones_like(y)
    P1_y = y
    P2_y = 0.5 * (3 * y**2 - 1)
    P3_y = 0.5 * (5 * y**3 - 3 * y)
    
    # Tensor product basis, 10 terms
    # Order:
    # 0: P0_x P0_y = 1
    # 1: P1_x P0_y = x
    # 2: P0_x P1_y = y
    # 3: P2_x P0_y = 0.5(3x^2 - 1)
    # 4: P0_x P2_y = 0.5(3y^2 - 1)
    # 5: P1_x P1_y = xy
    # 6: P3_x P0_y = 0.5(5x^3 - 3x)
    # 7: P0_x P3_y = 0.5(5y^3 - 3y)
    # 8: P2_x P1_y = 0.5(3x^2 - 1)y
    # 9: P1_x P2_y = x(0.5(3y^2 - 1))
    
    f = (params[0] * P0_x * P0_y +
         params[1] * P1_x * P0_y +
         params[2] * P0_x * P1_y +
         params[3] * P2_x * P0_y +
         params[4] * P0_x * P2_y +
         params[5] * P1_x * P1_y +
         params[6] * P3_x * P0_y +
         params[7] * P0_x * P3_y +
         params[8] * P2_x * P1_y +
         params[9] * P1_x * P2_y)
         
    return f
