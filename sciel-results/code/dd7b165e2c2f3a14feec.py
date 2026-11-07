"""
Strategy: Full second-order polynomial basis.

The model uses a complete polynomial expansion of degree 2 in three variables.
This basis consists of 10 terms:
  1 constant
  3 linear terms (x, y, z)
  3 squared terms (x^2, y^2, z^2)
  3 interaction terms (xy, xz, yz)

Total parameters: 10.

This approach is structurally different from the previous neural network (tanh)
and trigonometric bases. The key advantage is that the model is linear in the
parameters. The optimization problem (minimizing MSE) is therefore a convex
quadratic programming problem. This ensures that scipy.optimize.minimize with
BFGS will reliably converge to the global minimum of the training loss,
avoiding the local minima pitfalls of non-linear parameterized models like
neural networks or models with tunable frequencies.

If the hidden function is a polynomial of degree <= 2, this model will achieve
a near-perfect score. For higher-order or non-polynomial functions, it provides
a robust, stable approximation that is often superior to poorly-converged
non-linear models.
"""
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns predictions using a full second-order polynomial basis.
    
    Parameters:
    p[0]: constant term
    p[1]: coefficient for x
    p[2]: coefficient for y
    p[3]: coefficient for z
    p[4]: coefficient for x^2
    p[5]: coefficient for y^2
    p[6]: coefficient for z^2
    p[7]: coefficient for x*y
    p[8]: coefficient for x*z
    p[9]: coefficient for y*z
    """
    p = params.flatten()
    
    # Linear terms
    c0 = p[0]
    c1 = p[1]
    c2 = p[2]
    c3 = p[3]
    
    # Quadratic terms
    c4 = p[4]
    c5 = p[5]
    c6 = p[6]
    c7 = p[7]
    c8 = p[8]
    c9 = p[9]
    
    # Compute basis functions
    # We can compute this directly for speed
    return (c0 + 
            c1 * x + c2 * y + c3 * z + 
            c4 * x * x + c5 * y * y + c6 * z * z + 
            c7 * x * y + c8 * x * z + c9 * y * z)
