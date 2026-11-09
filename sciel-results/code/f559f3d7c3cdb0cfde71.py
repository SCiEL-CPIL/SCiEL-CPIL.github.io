"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression basis with 10 parameters using Legendre polynomials.
    
    Mixed-degree Legendre tensor product basis:
    - Complete through degree 2 (6 terms)
    - Axial degree-3 and degree-4 terms (4 terms)
    
    This sacrifices some cross-cubic flexibility for higher-order axial terms,
    which may better capture functions with strong x^n or y^n components.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Legendre polynomials up to degree 4
    P0_x = np.ones_like(x)
    P1_x = x
    P2_x = (3.0 * x * x - 1.0) / 2.0
    P3_x = (5.0 * x * x * x - 3.0 * x) / 2.0
    P4_x = (35.0 * x**4 - 30.0 * x**2 + 3.0) / 8.0
    
    P0_y = np.ones_like(y)
    P1_y = y
    P2_y = (3.0 * y * y - 1.0) / 2.0
    P3_y = (5.0 * y * y * y - 3.0 * y) / 2.0
    P4_y = (35.0 * y**4 - 30.0 * y**2 + 3.0) / 8.0
    
    # 10 basis terms
    b1 = P0_x * P0_y      # degree 0
    b2 = P1_x * P0_y      # degree 1
    b3 = P0_x * P1_y      # degree 1
    b4 = P2_x * P0_y      # degree 2
    b5 = P0_x * P2_y      # degree 2
    b6 = P1_x * P1_y      # degree 2
    b7 = P3_x * P0_y      # degree 3 (axial)
    b8 = P0_x * P3_y      # degree 3 (axial)
    b9 = P4_x * P0_y      # degree 4 (axial)
    b10 = P0_x * P4_y     # degree 4 (axial)
    
    return (params[0] * b1 + 
            params[1] * b2 + 
            params[2] * b3 + 
            params[3] * b4 + 
            params[4] * b5 + 
            params[5] * b6 + 
            params[6] * b7 + 
            params[7] * b8 + 
            params[9] * b9 + 
            params[8] * b10)
# EVOLVE-BLOCK-END
