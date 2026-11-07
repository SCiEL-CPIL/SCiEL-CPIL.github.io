import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation for symbolic regression with 2 inputs x, y.
    Uses a hybrid basis of low-order polynomials and RBFs centered on axes and origin.
    The 10 parameters correspond to the weights of these basis functions.
    """
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # 1. Polynomial Basis (Low order for stability and global structure)
    b1 = np.ones_like(x)       # Bias
    b2 = x                     # Linear x
    b3 = y                     # Linear y
    b4 = x * x                 # Quadratic x
    b5 = y * y                 # Quadratic y
    
    # 2. Radial Basis Functions (Gaussian kernels)
    # Centers placed at origin and mid-points of axes
    # Scale factor sigma_sq = 1.0 -> width ~ 0.7
    sigma_sq = 1.0
    
    # Center 1: (0, 0)
    r1_sq = x**2 + y**2
    b6 = np.exp(-sigma_sq * r1_sq)
    
    # Center 2: (0.5, 0)
    r2_sq = (x - 0.5)**2 + y**2
    b7 = np.exp(-sigma_sq * r2_sq)
    
    # Center 3: (-0.5, 0)
    r3_sq = (x + 0.5)**2 + y**2
    b8 = np.exp(-sigma_sq * r3_sq)
    
    # Center 4: (0, 0.5)
    r4_sq = x**2 + (y - 0.5)**2
    b9 = np.exp(-sigma_sq * r4_sq)
    
    # Center 5: (0, -0.5)
    r5_sq = x**2 + (y + 0.5)**2
    b10 = np.exp(-sigma_sq * r5_sq)
    
    # Combine linearly
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
            
    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
