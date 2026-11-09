import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Model: Full quadratic polynomial in x, y, z
    # f(x,y,z) = p[0] + p[1]*x + p[2]*y + p[3]*z + 
    #            p[4]*x^2 + p[5]*y^2 + p[6]*z^2 + 
    #            p[7]*xy + p[8]*xz + p[9]*yz
    
    # This basis has exactly 10 terms, matching the length of params.
    # It is linear in params, allowing BFGS to converge reliably.
    
    result = (p[0] + 
              p[1]*x + p[2]*y + p[3]*z + 
              p[4]*x**2 + p[5]*y**2 + p[6]*z**2 + 
              p[7]*x*y + p[8]*x*z + p[9]*y*z)
              
    return result
