"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Product of Sums (Multiplicative) Basis.
    
    The current quadratic polynomial (score 7.39) is likely underfitting a more complex function.
    A product of sums structure:
      P(x,y,z) = (a0 + a1*x + a2*y + a3*z) * (b0 + b1*x + b2*y + b3*z) * (c0 + c1*x + c2*y + c3*z)
    
    This allows the model to capture:
    1. Pure polynomials up to degree 3 (if constants are 0).
    2. Exponential-like growth (if linear terms dominate).
    3. Interactions that are not simply additive.
    
    With 10 parameters, we can form 2 full linear terms (using 8 params) and 2 scaling constants,
    or 3 linear terms sharing some structure, or a mix.
    
    Let's try:
    Term1 = p0 + p1*x + p2*y + p3*z
    Term2 = p4 + p5*x + p6*y + p7*z
    Term3 = p8 + p9*x
    
    Prediction = Term1 * Term2 * Term3
    
    This provides a rich, non-linear, non-polynomial (unless constrained) search space.
    """
    
    # Extract parameters
    p0, p1, p2, p3 = params[0], params[1], params[2], params[3]
    p4, p5, p6, p7 = params[4], params[5], params[6], params[7]
    p8, p9 = params[8], params[9]
    
    # Construct multiplicative terms
    # Using 3 factors to utilize all 10 params effectively.
    # Factor 1: Full linear in x,y,z
    f1 = p0 + p1 * x + p2 * y + p3 * z
    
    # Factor 2: Full linear in x,y,z
    f2 = p4 + p5 * x + p6 * y + p7 * z
    
    # Factor 3: Linear in x (to use remaining params)
    # Alternatively, we could make Factor 3 a constant scale p8 and use p9 elsewhere,
    # but a 3rd linear term adds more expressive power for interactions.
    f3 = p8 + p9 * x
    
    # Combine multiplicatively
    pred = f1 * f2 * f3
    
    # Safety check for non-finite values (though BFGS usually handles this, 
    # returning inf/nan breaks the evaluator)
    if not np.all(np.isfinite(pred)):
        # Return a large constant to penalize heavily but keep finite
        # Actually, the contract says "non-finite test predictions make the result invalid".
        # We should try to avoid this by construction or clipping if necessary,
        # but BFGS might step into bad regions.
        # For now, rely on the optimizer. If it fails, score is 0.
        # We can add a small epsilon regularization to inputs if needed, but let's keep it clean.
        pass
        
    return pred
# EVOLVE-BLOCK-END
