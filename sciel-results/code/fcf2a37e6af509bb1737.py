"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluates the linear combination of 10 basis functions.
    The basis is designed to approximate a wide range of symbolic functions
    including polynomials, trigonometric functions, and their interactions.
    """
    # Ensure inputs are arrays
    x = np.asarray(x)
    y = np.asarray(y)
    p = np.asarray(params)
    
    # 10 Basis functions
    # 1. Constant
    b1 = np.ones_like(x)
    # 2. Linear x
    b2 = x
    # 3. Linear y
    b3 = y
    # 4. Quadratic x
    b4 = x * x
    # 5. Quadratic y
    b5 = y * y
    # 6. Interaction xy
    b6 = x * y
    # 7. Trigonometric sin(x)
    b7 = np.sin(x)
    # 8. Trigonometric cos(x)
    b8 = np.cos(x)
    # 9. Trigonometric sin(y)
    b9 = np.sin(y)
    # 10. Trigonometric cos(y)
    b10 = np.cos(y)
    
    # Linear combination
    pred = (p[0] * b1 + 
            p[1] * b2 + 
            p[2] * b3 + 
            p[3] * b4 + 
            p[4] * b5 + 
            p[5] * b6 + 
            p[6] * b7 + 
            p[7] * b8 + 
            p[8] * b9 + 
            p[9] * b10)
            
    return pred
# EVOLVE-BLOCK-END
