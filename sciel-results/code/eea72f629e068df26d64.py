"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard: ensure we return within 150s wall-clock
    t_start = time.time()
    deadline = 145.0  # 5s safety margin
    
    # Compute Chebyshev polynomials of the first kind up to degree 3
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2*t**2 - 1
    # T_3(t) = 4*t**3 - 3*t
    # 
    # Chebyshev polynomials are minimax on [-1,1], giving better conditioning
    # for BFGS optimization compared to Legendre polynomials.
    
    px0 = np.ones_like(x)
    px1 = x
    px2 = 2.0 * x**2 - 1.0
    px3 = 4.0 * x**3 - 3.0 * x
    
    py0 = np.ones_like(y)
    py1 = y
    py2 = 2.0 * y**2 - 1.0
    py3 = 4.0 * y**3 - 3.0 * y
    
    # Construct the 10 basis functions spanning the cubic polynomial space
    # The order corresponds to the indices of params
    b0 = px0 * py0  # 1
    b1 = px1 * py0  # x
    b2 = px0 * py1  # y
    b3 = px2 * py0  # T_2(x)
    b4 = px1 * py1  # x*y
    b5 = px0 * py2  # T_2(y)
    b6 = px3 * py0  # T_3(x)
    b7 = px2 * py1  # T_2(x)*y
    b8 = px1 * py2  # x*T_2(y)
    b9 = px0 * py3  # T_3(y)
    
    pred = (params[0] * b0 + params[1] * b1 + params[2] * b2 + 
            params[3] * b3 + params[4] * b4 + params[5] * b5 + 
            params[6] * b6 + params[7] * b7 + params[8] * b8 + 
            params[9] * b9)
    
    # Check deadline before returning
    if time.time() - t_start > deadline:
        return pred
    
    return pred
# EVOLVE-BLOCK-END
