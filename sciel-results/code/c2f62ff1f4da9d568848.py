"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a hybrid basis of polynomials and tanh nonlinearities.
    This basis aims to capture both polynomial trends and non-linear saturation effects,
    which are common in smooth target functions.
    """
    start_time = time.time()
    
    try:
        # Ensure inputs are float arrays
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Basis Terms
        # 1. Constant
        term0 = np.ones_like(x)
        
        # 2. Linear x
        term1 = x
        
        # 3. Linear y
        term2 = y
        
        # 4. Quadratic x
        term3 = x**2
        
        # 5. Quadratic y
        term4 = y**2
        
        # 6. Interaction xy
        term5 = x * y
        
        # 7. Non-linear x
        term6 = np.tanh(x)
        
        # 8. Non-linear y
        term7 = np.tanh(y)
        
        # 9. Non-linear sum
        term8 = np.tanh(x + y)
        
        # 10. Non-linear diff
        term9 = np.tanh(x - y)
        
        # Combine linearly
        prediction = (params[0] * term0 + 
                      params[1] * term1 + 
                      params[2] * term2 + 
                      params[3] * term3 + 
                      params[4] * term4 + 
                      params[5] * term5 + 
                      params[6] * term6 + 
                      params[7] * term7 + 
                      params[8] * term8 + 
                      params[9] * term9)
        
        # Check for non-finite values
        if not np.all(np.isfinite(prediction)):
             return np.zeros_like(x)
             
        return prediction
        
    except Exception:
        return np.zeros_like(x)
# EVOLVE-BLOCK-END
