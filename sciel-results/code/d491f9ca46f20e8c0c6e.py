import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a rational basis.
    Basis: 1, x, y, z, x/y, x/z, y/x, y/z, z/x, z/y
    This allows the optimizer to fit functions that can be approximated by rational expressions.
    """
    p = params
    
    # Ensure inputs are float arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    
    # Basis terms
    t0 = np.ones_like(x)
    t1 = x
    t2 = y
    t3 = z
    t4 = x / y
    t5 = x / z
    t6 = y / x
    t7 = y / z
    t8 = z / x
    t9 = z / y
    
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
