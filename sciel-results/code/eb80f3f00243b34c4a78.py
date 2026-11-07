import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map inputs from [0.5, 2] to [-1, 1] for Chebyshev polynomials
    # t = (4u - 5) / 3
    tx = (4.0 * x - 5.0) / 3.0
    ty = (4.0 * y - 5.0) / 3.0
    tz = (4.0 * z - 5.0) / 3.0
    
    # Chebyshev polynomials of the first kind
    # T0 = 1
    # T1 = t
    # T2 = 2t^2 - 1
    # T3 = 4t^3 - 3t
    
    # Compute terms for x
    t1_x = tx
    t2_x = 2.0 * tx * tx - 1.0
    t3_x = 4.0 * tx * tx * tx - 3.0 * tx
    
    # Compute terms for y
    t1_y = ty
    t2_y = 2.0 * ty * ty - 1.0
    t3_y = 4.0 * ty * ty * ty - 3.0 * ty
    
    # Compute terms for z
    t1_z = tz
    t2_z = 2.0 * tz * tz - 1.0
    t3_z = 4.0 * tz * tz * tz - 3.0 * tz
    
    p = params
    
    # Basis:
    # 0: Constant
    # 1: T1(x)
    # 2: T2(x)
    # 3: T3(x)
    # 4: T1(y)
    # 5: T2(y)
    # 6: T3(y)
    # 7: T1(z)
    # 8: T2(z)
    # 9: T3(z)
    
    return (p[0] + 
            p[1] * t1_x + p[2] * t2_x + p[3] * t3_x + 
            p[4] * t1_y + p[5] * t2_y + p[6] * t3_y + 
            p[7] * t1_z + p[8] * t2_z + p[9] * t3_z)
