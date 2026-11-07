"""
Symbolic regression using a fixed linear basis of polynomials and RBFs.
This makes the problem linear in the parameters, allowing BFGS to converge
to the global optimum for the chosen basis efficiently.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of a fixed basis of 10 functions.
    
    Basis functions:
    1. Constant: 1
    2. Linear: x
    3. Quadratic: x^2
    4. Cubic: x^3
    5. Quartic: x^4
    6. Quintic: x^5
    7-10. Gaussian RBFs with fixed centers at 1.0, 2.5, 4.0, 5.5 and width sigma=0.8
    
    Prediction: y = sum_{i=0}^{9} params[i] * basis_i(x)
    """
    # Safety deadline check (though this function is very fast, it's good practice)
    start_time = time.time()
    if start_time > 110: # Hard stop if we are already late (unlikely)
        return np.zeros_like(x, dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure params are finite
    if not np.all(np.isfinite(p)):
        p = np.where(np.isfinite(p), p, 0.0)
        
    # Build the basis matrix Phi of shape (N, 10)
    # N is the number of points
    N = x.shape[0]
    Phi = np.empty((N, 10), dtype=np.float64)
    
    # 1-6: Polynomials
    # Using np.power or explicit multiplication for speed
    Phi[:, 0] = 1.0
    Phi[:, 1] = x
    Phi[:, 2] = x * x
    Phi[:, 3] = Phi[:, 2] * x
    Phi[:, 4] = Phi[:, 3] * x
    Phi[:, 5] = Phi[:, 4] * x
    
    # 7-10: Gaussian RBFs
    # Centers chosen to cover [0.1, 6]
    centers = np.array([1.0, 2.5, 4.0, 5.5])
    sigma = 0.8
    sigma_sq = sigma * sigma
    
    for i, c in enumerate(centers):
        # exp(-((x - c)^2) / (2 * sigma^2))
        diff = x - c
        Phi[:, 6 + i] = np.exp(-0.5 * (diff * diff) / sigma_sq)
        
    # Prediction is linear combination
    prediction = Phi @ p
    
    # Check for non-finite values (should not happen with this basis, but safe)
    if not np.all(np.isfinite(prediction)):
        prediction = np.where(np.isfinite(prediction), prediction, 0.0)
        
    return prediction
# EVOLVE-BLOCK-END
