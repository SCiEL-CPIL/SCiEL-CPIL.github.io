import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Basis: 4 Polynomial Terms + 6 Gaussian RBFs.
    
    The previous quadratic basis (score 7.39) failed, suggesting the target
    function is not well-approximated by a quadratic polynomial. 
    Since the evaluator uses BFGS to fit 10 parameters, and the basis functions
    are fixed (linear in params), the optimization is a linear least-squares
    problem. BFGS will converge to the global minimum for this linear model.
    
    We use a richer basis that includes local flexibility via Radial Basis
    Functions (RBFs) to capture non-polynomial structures.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.shape != (10,):
        p = np.resize(p, 10)
        
    # Input scaling: map [0.5, 2.0] to [0, 1] for better RBF behavior?
    # Or keep original scale for RBF centers.
    # Let's use original scale for centers, but scale inputs for polynomial terms?
    # Actually, for RBFs, the scale of the input matters for the width.
    # Domain size is 1.5. A width of 0.5 covers a good portion.
    
    # Basis 0-3: Polynomial (Constant, x, y, z)
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = z
    
    # Basis 4-9: Gaussian RBFs
    # Centers chosen to cover the corners and center of the domain [0.5, 2.0]^3
    centers = np.array([
        [0.5, 0.5, 0.5],
        [2.0, 0.5, 0.5],
        [0.5, 2.0, 0.5],
        [0.5, 0.5, 2.0],
        [2.0, 2.0, 0.5],
        [1.25, 1.25, 1.25]
    ])
    
    sigma = 0.4  # Width parameter
    
    # Calculate squared distances for each center
    # d2 = (x-cx)^2 + (y-cy)^2 + (z-cz)^2
    d2_0 = (x - centers[0, 0])**2 + (y - centers[0, 1])**2 + (z - centers[0, 2])**2
    d2_1 = (x - centers[1, 0])**2 + (y - centers[1, 1])**2 + (z - centers[1, 2])**2
    d2_2 = (x - centers[2, 0])**2 + (y - centers[2, 1])**2 + (z - centers[2, 2])**2
    d2_3 = (x - centers[3, 0])**2 + (y - centers[3, 1])**2 + (z - centers[3, 2])**2
    d2_4 = (x - centers[4, 0])**2 + (y - centers[4, 1])**2 + (z - centers[4, 2])**2
    d2_5 = (x - centers[5, 0])**2 + (y - centers[5, 1])**2 + (z - centers[5, 2])**2
    
    b4 = np.exp(-d2_0 / (2 * sigma**2))
    b5 = np.exp(-d2_1 / (2 * sigma**2))
    b6 = np.exp(-d2_2 / (2 * sigma**2))
    b7 = np.exp(-d2_3 / (2 * sigma**2))
    b8 = np.exp(-d2_4 / (2 * sigma**2))
    b9 = np.exp(-d2_5 / (2 * sigma**2))
    
    B = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    result = B @ p
    
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
