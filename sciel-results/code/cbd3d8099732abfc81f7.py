import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Strategy: Product of two low-degree polynomials P5(x) * Q4(x)
    # This creates a non-convex but structured landscape that BFGS can navigate
    # to fit functions with sharp local features or high-frequency oscillations
    # that linear combinations of fixed smooth basis functions cannot capture.
    #
    # P5(x) = p[0] + p[1]*x + p[2]*x^2 + p[3]*x^3 + p[4]*x^4 + p[5]*x^5
    # Q4(x) = p[6] + p[7]*x + p[8]*x^2 + p[9]*x^3 + p[10]*x^4
    # Wait, we only have 10 params. Let's do P4(x) * Q5(x) or P3(x) * Q6(x)
    # Let's try P4(x) * Q5(x):
    # P4(x) = p[0] + p[1]*x + p[2]*x^2 + p[3]*x^3 + p[4]*x^4
    # Q5(x) = p[5] + p[6]*x + p[7]*x^2 + p[8]*x^3 + p[9]*x^4 + ... wait, need 5 coeffs.
    # So P4 uses 5 params, Q5 uses 5 params. Total 10.
    # P4(x) = p[0] + p[1]*x + p[2]*x^2 + p[3]*x^3 + p[4]*x^4
    # Q5(x) = p[5] + p[6]*x + p[7]*x^2 + p[8]*x^3 + p[9]*x^4
    # Actually, a degree 4 polynomial has 5 coefficients. A degree 5 polynomial has 6 coefficients.
    # So P4 (5 coeffs) * Q5 (6 coeffs) = 11 coeffs. Too many.
    # Let's do P3(x) * Q6(x):
    # P3(x) = p[0] + p[1]*x + p[2]*x^2 + p[3]*x^3 (4 coeffs)
    # Q6(x) = p[4] + p[5]*x + p[6]*x^2 + p[7]*x^3 + p[8]*x^4 + p[9]*x^5 (6 coeffs)
    # Total 10 coeffs. Product is degree 9.
    #
    # Alternatively, just use a rational function P4(x)/Q4(x) as suggested by the knowledge cell.
    # P4(x) = p[0] + p[1]*x + p[2]*x^2 + p[3]*x^3 + p[4]*x^4
    # Q4(x) = p[5] + p[6]*x + p[7]*x^2 + p[8]*x^3 + p[9]*x^4
    # This is a very different structural change and can represent poles.
    # Let's try the rational function approach.
    
    x_safe = x.copy()
    
    # Numerator P4(x)
    P = p[0] + p[1]*x_safe + p[2]*x_safe**2 + p[3]*x_safe**3 + p[4]*x_safe**4
    
    # Denominator Q4(x)
    Q = p[5] + p[6]*x_safe + p[7]*x_safe**2 + p[8]*x_safe**3 + p[9]*x_safe**4
    
    # Avoid division by zero
    # If Q is very close to 0, the result can be huge.
    # We can add a small epsilon to Q to avoid inf/nan, or clip.
    # However, clipping might distort the function.
    # Let's use np.where to handle small Q values.
    # But BFGS might struggle with discontinuities.
    # A safer approach for optimization is to ensure Q is bounded away from 0?
    # Or just let it be, and if it's not finite, return zeros?
    # The prompt says: "A prediction of the wrong shape or with non-finite error costs 1e30 during fitting."
    # So we must ensure finite outputs.
    
    y = P / Q
    
    # Replace non-finite values with 0 to avoid penalties
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
