"""Symbolic regression with 1 input using a rational polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points (200 train or 200 test)
    params: array of 10 parameters
    Returns: predicted values
    
    Strategy:
    Use a basis of functions phi_i(x) = x^i / (1 + x^2).
    This basis is linear in the parameters, allowing BFGS to solve it
    efficiently as a least-squares problem.
    It is particularly strong for functions with rational structure,
    such as Cauchy distributions or functions with asymptotic behavior
    that pure polynomials approximate poorly.
    The denominator (1 + x^2) is always positive, ensuring numerical stability.
    """
    n = len(x)
    
    # Precompute the denominator
    # x is in [-2, 2], so x^2 is in [0, 4]. 1+x^2 is in [1, 5].
    # No risk of division by zero.
    denom = 1.0 + x**2
    
    # Build the basis and combine
    # We use 10 terms: i from 0 to 9
    # y = sum(params[i] * (x**i / denom))
    
    # To avoid computing x**i repeatedly in a loop which might be slow,
    # we can use vectorized operations or a simple loop.
    # Given n=200 and 10 terms, a loop is fine and clear.
    
    y = np.zeros(n, dtype=np.float64)
    
    # Start with x^0 / denom
    term = np.ones(n)
    for i in range(10):
        y += params[i] * (term / denom)
        # Update term to x^(i+1) for next iteration
        if i < 9:
            term = term * x
            
    return y
# EVOLVE-BLOCK-END
