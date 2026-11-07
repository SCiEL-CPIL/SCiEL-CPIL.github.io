import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a mixed basis of rational, product, and ratio terms.
    Basis: 1, x, y, z, x*y, x*z, y*z, x/y, y/z, z/x
    This spans a richer space than pure linear or pure rational, capturing multiplicative
    interactions and key ratios that often appear in physical/engineering formulas.
    """
    p = np.asarray(params, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Basis terms
    t0 = np.ones_like(x)
    t1 = x
    t2 = y
    t3 = z
    t4 = x * y
    t5 = x * z
    t6 = y * z
    t7 = x / y
    t8 = y / z
    t9 = z / x
    
    # Linear combination
    result = (p[0] * t0 + 
              p[1] * t1 + 
              p[2] * t2 + 
              p[3] * t3 + 
              p[4] * t4 + 
              p[5] * t5 + 
              p[6] * t6 + 
              p[7] * t7 + 
              p[8] * t8 + 
              p[9] * t9)
              
    return result
