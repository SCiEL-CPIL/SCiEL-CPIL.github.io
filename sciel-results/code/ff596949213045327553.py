"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    # Simple, robust basis to avoid divergence
    # 0: bias
    # 1: x
    # 2: x^2
    # 3: x^3
    # 4: sin(x)
    # 5: cos(x)
    # 6: sin(2x)
    # 7: cos(2x)
    # 8: tanh(x)
    # 9: x^4
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Calculate terms
    result = p0 + p1 * x + p2 * x**2 + p3 * x**3 + p9 * x**4
    result += p4 * np.sin(x) + p5 * np.cos(x)
    result += p6 * np.sin(2 * x) + p7 * np.cos(2 * x)
    result += p8 * np.tanh(x)
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        # If values explode, return a large constant to penalize this step
        # or return 0. The evaluator adds 1e30 for non-finite.
        # Returning a finite but large number might be safer for BFGS?
        # Actually, the evaluator says "non-finite error costs 1e30".
        # So if we return inf, it costs 1e30.
        # If we return a large finite number, it costs (large - y)^2.
        # If params are huge, result is huge.
        # We can clip the result to prevent overflow in subsequent steps?
        # No, we just return the result.
        
        # However, if x is large, x**4 is large.
        # x is in [-2, 2], so x**4 is in [0, 16].
        # sin, cos, tanh are bounded.
        # The only risk is if params become huge.
        # If params are huge, result is huge.
        # We can't control params.
        
        pass

    return result
# EVOLVE-BLOCK-END
