import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Compute Legendre polynomials up to order 4 for x and y
    # P0 = 1
    # P1 = x
    # P2 = 0.5 * (3x^2 - 1)
    # P3 = 0.5 * (5x^3 - 3x)
    # P4 = 0.125 * (35x^4 - 30x^2 + 3)
    
    px0 = np.ones_like(x)
    px1 = x
    px2 = 0.5 * (3 * x**2 - 1)
    px3 = 0.5 * (5 * x**3 - 3 * x)
    px4 = 0.125 * (35 * x**4 - 30 * x**2 + 3)
    
    py0 = np.ones_like(y)
    py1 = y
    py2 = 0.5 * (3 * y**2 - 1)
    py3 = 0.5 * (5 * y**3 - 3 * y)
    py4 = 0.125 * (35 * y**4 - 30 * y**2 + 3)
    
    # Construct basis columns with higher-order terms
    # 1. P0(x)P0(y)
    # 2. P1(x)P0(y)
    # 3. P0(x)P1(y)
    # 4. P2(x)P0(y)
    # 5. P0(x)P2(y)
    # 6. P1(x)P1(y)
    # 7. P3(x)P0(y)
    # 8. P0(x)P3(y)
    # 9. P2(x)P1(y)
    # 10. P1(x)P2(y)
    
    b1 = px0 * py0
    b2 = px1 * py0
    b3 = px0 * py1
    b4 = px2 * py0
    b5 = px0 * py2
    b6 = px1 * py1
    b7 = px3 * py0
    b8 = px0 * py3
    b9 = px2 * py1
    b10 = px1 * py2
    
    # Stack into matrix
    basis = np.column_stack((b1, b2, b3, b4, b5, b6, b7, b8, b9, b10))
    
    # Prediction
    return basis @ params
