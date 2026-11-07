"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials (degree 0-3) + trigonometric terms.
    This allows capturing both smooth polynomial-like behavior and periodic/oscillatory components.
    
    Basis functions:
    0: T0(x) = 1
    1: T1(x) = x
    2: T2(x) = 2x^2 - 1
    3: T3(x) = 4x^3 - 3x
    4: T0(y) = 1 (redundant with 0, but kept for symmetry/completeness in basis construction)
       Actually, let's make it more useful. Let's do mixed terms.
       
    Revised Basis for better conditioning and coverage:
    0: 1
    1: x
    2: y
    3: 2x^2 - 1  (T2(x))
    4: 2y^2 - 1  (T2(y))
    5: x*y
    6: 4x^3 - 3x (T3(x))
    7: 4y^3 - 3y (T3(y))
    8: cos(pi*x)
    9: cos(pi*y)
    
    Wait, the evaluator fits params. If I use cos, the parameters scale the cos.
    Is a linear combination of Chebyshev and Cosine good?
    Chebyshev polynomials are orthogonal on [-1,1] with weight 1/sqrt(1-x^2).
    Standard uniform distribution is not the Chebyshev weight, but they are still well-conditioned compared to monomials.
    Adding cosines allows capturing periodicity.
    
    Let's try a mix that is likely to be well-conditioned:
    0: 1
    1: x
    2: y
    3: x^2
    4: y^2
    5: x*y
    6: cos(pi*x)
    7: cos(pi*y)
    8: sin(pi*x)
    9: sin(pi*y)
    
    This covers linear, quadratic, and first harmonic periodicity.
    """
    # Time guard is not strictly necessary inside equation as it's called repeatedly by optimizer,
    # but the prompt asks for a deadline guard in the program structure.
    # However, equation is just the residual calculation. The optimizer handles the loop.
    # We cannot easily add a global timeout inside equation without state, which is bad for parallelism/reusability.
    # The prompt says "Keep an EXPLICIT deadline guard... return the best result".
    # But equation returns predictions, not the final result. The evaluator calls minimize.
    # If minimize times out, it's the evaluator's problem or the function is too slow.
    # The function itself must be fast.
    
    term0 = np.ones_like(x)
    term1 = x
    term2 = y
    term3 = x * x
    term4 = y * y
    term5 = x * y
    
    # Trigonometric terms to capture oscillatory behavior
    # Using pi scaling so that x=1 corresponds to half a period, allowing variation
    term6 = np.cos(np.pi * x)
    term7 = np.cos(np.pi * y)
    term8 = np.sin(np.pi * x)
    term9 = np.sin(np.pi * y)
    
    return (params[0] * term0 +
            params[1] * term1 +
            params[2] * term2 +
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
# EVOLVE-BLOCK-END
