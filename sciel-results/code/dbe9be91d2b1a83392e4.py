"""
Strategy: Hybrid Polynomial Basis (Quadratic + Selective Cubic).

The previous best (Quadratic Polynomial, score ~7.39) perfectly fits any quadratic function
but fails on higher-order terms. The Gaussian RBF model (score ~253) failed due to
non-convex optimization issues.

This program uses a linear-in-parameters basis that extends the quadratic space
by including specific cubic terms: $x^3, y^3, z^3$ is too many (3 extra).
We have 10 slots.
Standard Quadratic: 1, x, y, z, x^2, y^2, z^2, xy, xz, yz (10 terms).

We propose a basis that trades some quadratic cross-terms for cubic diagonal terms,
assuming that higher-order diagonal behavior (e.g., $x^3$) is more likely in symbolic
regression tasks than specific cross-terms like $xz$ if the function is separable or
sum-of-powers.
However, $xyz$ is a very common symbolic term.

Basis Chosen:
1.  1
2.  x
3.  y
4.  z
5.  x^2
6.  y^2
7.  z^2
8.  xyz
9.  x^3
10. y^3

This covers:
- Constant, Linear
- Quadratic Diagonals
- Main Cubic Interaction (xyz)
- Cubic Diagonals for x and y

If the target is $x^3 + y^3 + z^3$, we fit x and y perfectly, and z is approximated by
lower order terms (error).
If the target is $x^2 + y^2 + z^2 + xyz$, we fit everything perfectly.
If the target is $x y z + 1$, we fit perfectly.

This linear model ensures BFGS converges to the global optimum of this specific subspace.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the linear combination of the basis functions.
    """
    # Extract parameters
    c0 = params[0]
    cx = params[1]
    cy = params[2]
    cz = params[3]
    cx2 = params[4]
    cy2 = params[5]
    cz2 = params[6]
    cxyz = params[7]
    cx3 = params[8]
    cy3 = params[9]
    
    # Compute basis terms
    term_0 = c0
    term_x = cx * x
    term_y = cy * y
    term_z = cz * z
    term_x2 = cx2 * (x * x)
    term_y2 = cy2 * (y * y)
    term_z2 = cz2 * (z * z)
    term_xyz = cxyz * (x * y * z)
    term_x3 = cx3 * (x * x * x)
    term_y3 = cy3 * (y * y * y)
    
    # Sum
    result = term_0 + term_x + term_y + term_z + term_x2 + term_y2 + term_z2 + term_xyz + term_x3 + term_y3
    
    return result
# EVOLVE-BLOCK-END
