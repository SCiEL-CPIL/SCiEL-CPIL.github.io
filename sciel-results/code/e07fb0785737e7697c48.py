"""
Strategy: Cubic B-Spline Representation
Instead of a fixed basis of polynomials and RBFs, we use the 10 parameters
directly as control points for a cubic B-spline. The function computes the
B-spline basis functions for the input x and returns the linear combination
weighted by the parameters. This allows BFGS to fit a smooth, flexible curve
that can adapt to local features of the target function.
"""
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a cubic B-spline representation.
    
    The 10 parameters are the control points of the B-spline.
    The domain is [-3, 3]. We use a knot vector that covers this domain.
    With 10 control points and cubic degree (k=3), we need 10 + 3 + 1 = 14 knots.
    We use uniform knots in [-3, 3] with padding for the boundary.
    
    To make the spline cover the entire domain [-3, 3] with 10 control points,
    we define the knot vector as:
    k = [-3, -2.4, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4, 3, 3, 3, 3]
    Wait, standard B-spline definition:
    Number of knots = n + k + 1, where n is number of control points (10), k is degree (3).
    Total knots = 14.
    Interior knots should be in [-3, 3].
    Let's use a uniform knot vector that allows the spline to be defined on [-3, 3].
    A common choice is to have the knots span the domain with multiplicity k+1 at the ends.
    Knots: [-3, -3, -3, -3, -2.4, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4, 3, 3, 3, 3]
    This has 17 knots. n+k+1 = 10+3+1 = 14. Mismatch.
    
    Let's use a simpler approach: De Boor's algorithm or direct evaluation.
    Actually, let's just use a fixed set of basis functions that are B-splines.
    We can compute the B-spline basis functions N_{i,3}(x) for i=0..9.
    
    Let's define the knot vector U = [-3, -2.4, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4, 3]
    This has 11 knots. For cubic B-splines with 10 control points, we need 14 knots.
    Let's use U = [-3, -3, -3, -3, -2.4, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4, 3, 3, 3, 3]
    This is 17 knots. n=10, k=3. n+k+1 = 14. Still mismatch.
    
    Correct relation: Number of knots = n + k + 1.
    If n=10, k=3, knots = 14.
    Let's use U = [-3, -2.4, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4, 3, 3, 3, 3]
    This has 14 knots.
    The domain of the spline is [U[k], U[n]] = [U[3], U[10]] = [-1.2, 1.8].
    This does not cover [-3, 3].
    
    To cover [-3, 3] with 10 control points and cubic B-splines:
    We need the first knot U[k] = -3 and the last relevant knot U[n] = 3.
    U = [u0, u1, ..., u13]
    u3 = -3, u10 = 3.
    Let's choose uniform interior knots.
    u0=-3, u1=-3, u2=-3, u3=-3
    u4=-2.4, u5=-1.8, u6=-1.2, u7=-0.6, u8=0, u9=0.6, u10=1.2
    Wait, u10 must be 3.
    Let's adjust.
    u3 = -3
    u10 = 3
    The interval is 6. There are 10-3=7 intervals between u3 and u10? No, indices 3 to 10 is 7 steps.
    Let's just use a simpler basis: Lagrange polynomials at 10 evenly spaced points.
    This is structurally different from Chebyshev (which is global orthogonal) and RBFs (which are local but not polynomial).
    Lagrange interpolation at 10 points in [-3, 3] is a global polynomial of degree 9.
    It is linear in parameters.
    It can fit any function sampled at those 10 points exactly, but may oscillate (Runge's phenomenon).
    However, BFGS will find the optimal weights.
    
    Let's try **Lagrange Polynomials** at 10 evenly spaced nodes.
    Nodes: x_i = -3 + i * (6/9) for i=0..9.
    Basis function L_i(x) = prod_{j != i} (x - x_j) / (x_i - x_j).
    This is a degree 9 polynomial.
    It is a very different structural family from Chebyshev (orthogonal) and RBFs (exponential).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    
    # 10 evenly spaced nodes in [-3, 3]
    nodes = np.linspace(-3, 3, 10)
    
    # Compute Lagrange basis functions
    # basis[i] is L_i(x)
    basis = np.zeros((10, len(x)))
    
    for i in range(10):
        # L_i(x) = prod_{j != i} (x - x_j) / (x_i - x_j)
        Li = np.ones_like(x)
        for j in range(10):
            if i != j:
                Li *= (x - nodes[j]) / (nodes[i] - nodes[j])
        basis[i] = Li
        
    # Linear combination
    return basis.T @ params
