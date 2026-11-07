import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Legendre polynomials on [-1, 1]
    # P0(t) = 1
    # P1(t) = t
    # P2(t) = 1.5 * t^2 - 0.5
    # P3(t) = 2.5 * t^3 - 1.5 * t
    
    Px0 = np.ones_like(x)
    Px1 = x
    Px2 = 1.5 * x**2 - 0.5
    Px3 = 2.5 * x**3 - 1.5 * x
    
    Py0 = np.ones_like(y)
    Py1 = y
    Py2 = 1.5 * y**2 - 0.5
    Py3 = 2.5 * y**3 - 1.5 * y
    
    # Construct the 10 basis functions corresponding to the complete polynomial space of degree <= 3
    # We map the parameters to these orthogonal basis functions.
    # The order doesn't matter for the fit, as long as we cover the space.
    
    b0 = Px0 * Py0      # 1
    b1 = Px1 * Py0      # x
    b2 = Px0 * Py1      # y
    b3 = Px2 * Py0      # 1.5x^2 - 0.5
    b4 = Px0 * Py2      # 1.5y^2 - 0.5
    b5 = Px1 * Py1      # xy
    b6 = Px3 * Py0      # 2.5x^3 - 1.5x
    b7 = Px0 * Py3      # 2.5y^3 - 1.5y
    b8 = Px2 * Py1      # (1.5x^2 - 0.5)y
    b9 = Px1 * Py2      # x(1.5y^2 - 0.5)
    
    return (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 + 
            p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
