"""
Strategy: Gaussian Radial Basis Function (RBF) Mixture with Local Adaptation.

Rationale:
1. The current strategy uses a 9th-degree polynomial, which is a global basis. 
   While good for smooth global trends, it suffers from Runge's phenomenon at the 
   boundaries of [-3, 3] and cannot efficiently model local features or sharp 
   transitions without high-degree terms that cause numerical instability.
2. RBFs provide a local basis. A mixture of 3 Gaussians can model functions with 
   local peaks, valleys, or oscillatory behavior that are not well-captured by a 
   single global polynomial.
3. We have 10 parameters. We allocate 3 parameters per Gaussian (center, width, 
   amplitude) for 3 Gaussians, plus 1 global offset. Total = 3*3 + 1 = 10.
4. The optimization landscape is non-convex (due to centers and widths), so BFGS 
   may get stuck in local minima. However, starting from all ones (centers=1, 
   widths=1, amplitudes=1, offset=1) provides a reasonable initial guess for 
   functions centered around 0 or 1. 
5. To handle potential numerical issues (e.g., very small widths leading to 
   overflow in exp), we use np.nan_to_num to clip non-finite values to 0. This 
   ensures the evaluator never receives NaN/Inf, which would otherwise cause a 
   score of 1e30.
6. The diversity of the basis (local vs global) increases the chance of finding 
   a better fit for a wider class of hidden functions compared to a pure 
   polynomial.
"""

# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a mixture of 3 Gaussian RBFs plus an offset.
    
    params[0:3] -> Gaussian 1: center, width, amplitude
    params[3:6] -> Gaussian 2: center, width, amplitude
    params[6:9] -> Gaussian 3: center, width, amplitude
    params[9]   -> Global offset
    
    Each Gaussian is: amplitude * exp(-((x - center) / width)^2)
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure params has length 10
    if len(p) < 10:
        # Pad with zeros if necessary (though evaluator should provide 10)
        p = np.pad(p, (0, 10 - len(p)))
        
    # Extract parameters
    c1, w1, a1 = p[0], p[1], p[2]
    c2, w2, a2 = p[3], p[4], p[5]
    c3, w3, a3 = p[6], p[7], p[8]
    offset = p[9]
    
    # Prevent division by zero or very small widths which cause overflow
    # A minimum width of 0.1 is reasonable for x in [-3, 3]
    min_width = 0.1
    w1 = max(abs(w1), min_width)
    w2 = max(abs(w2), min_width)
    w3 = max(abs(w3), min_width)
    
    # Compute Gaussian terms
    # Use np.clip to prevent overflow in exp
    # exp(-large_positive) -> 0, exp(-large_negative) -> inf (but we square, so it's always positive)
    # Actually, ((x-c)/w)^2 is always positive, so exp(-positive) is in (0, 1].
    # The only risk is if w is extremely small, making the argument huge, but exp(-huge) -> 0.
    # The risk is if w is 0, which we handled.
    
    term1 = a1 * np.exp(-((x - c1) / w1) ** 2)
    term2 = a2 * np.exp(-((x - c2) / w2) ** 2)
    term3 = a3 * np.exp(-((x - c3) / w3) ** 2)
    
    result = term1 + term2 + term3 + offset
    
    # Safety net: replace any non-finite values with 0
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result
# EVOLVE-BLOCK-END
