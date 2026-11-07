import numpy as np
import time
from scipy.optimize import minimize

# Module-level cache for fitted parameters
_best_params = None
_fit_done = False

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich basis function approach for symbolic regression.
    
    Uses a combination of:
    - Polynomial terms (up to degree 3, but limited to keep params at 10)
    - Gaussian RBFs at strategic centers
    - Inverse/rational terms
    
    The evaluator fits 10 parameters via BFGS. We design a basis that is
    flexible enough to capture various function structures.
    """
    global _best_params, _fit_done
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.shape != (10,):
        p = np.resize(p, 10)
    
    # Build the basis matrix B (n_samples x 10)
    # We use a diverse set of basis functions:
    # b0: constant
    # b1: x
    # b2: y
    # b3: z
    # b4: x*y
    # b5: y*z
    # b6: x*z
    # b7: 1/x (inverse term for rational functions)
    # b8: 1/y
    # b9: 1/z
    
    # Actually, let me think more carefully. The score of 3.6761 with the
    # current approach (4 poly + 6 RBFs) suggests we need something different.
    # The inspirations with score 0.1701 used hybrid polynomial + RBF.
    # Let me try a different mix: more polynomial terms and fewer RBFs,
    # or a completely different structure.
    
    # Let's try: 
    # b0: 1
    # b1: x
    # b2: y  
    # b3: z
    # b4: x^2
    # b5: y^2
    # b6: z^2
    # b7: x*y
    # b8: y*z
    # b9: x*z
    
    # This is a full quadratic polynomial in 3 variables (10 terms).
    # If the target is a quadratic, this will fit perfectly.
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = z
    b4 = x**2
    b5 = y**2
    b6 = z**2
    b7 = x * y
    b8 = y * z
    b9 = x * z
    
    B = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    result = B @ p
    
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
