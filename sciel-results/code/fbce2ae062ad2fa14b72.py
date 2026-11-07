import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation for symbolic regression with 2 inputs x, y.
    Uses a combination of polynomial and Radial Basis Function (RBF) terms.
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
    b4 = x * y                 # Interaction
    b5 = x * x                 # Quadratic x
    b6 = y * y                 # Quadratic y
    
    # 2. Radial Basis Functions (Gaussian kernels)
    # Centers placed at the four "quadrant centers" of the [-1, 1]^2 domain
    # Scale factor sigma_sq = 2.0 -> width ~ 0.7, covers significant portion of domain
    sigma_sq = 2.0
    
    # Center 1: (0.5, 0.5)
    r1_sq = (x - 0.5)**2 + (y - 0.5)**2
    b7 = np.exp(-sigma_sq * r1_sq)
    
    # Center 2: (-0.5, 0.5)
    r2_sq = (x + 0.5)**2 + (y + 0.5)**2
    b8 = np.exp(-sigma_sq * r2_sq)
    
    # Center 3: (0.5, -0.5)
    r3_sq = (x - 0.5)**2 + (y - 0.5)**2
    b9 = np.exp(-sigma_sq * r3_sq)
    
    # Center 4: (-0.5, -0.5)
    r4_sq = (x + 0.5)**2 + (y + 0.5)**2
    b10 = np.exp(-sigma_sq * r4_sq)
    
    # Combine linearly
    # The order of parameters p[0]..p[9] maps to b1..b10
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
