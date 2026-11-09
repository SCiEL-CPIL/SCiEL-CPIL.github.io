import numpy as np
from scipy.optimize import minimize_scalar
from math import pi, cos, sin

def construct():
    # The 12 smallest positive zeros of J_{1/2}(x) are (k*pi)^2 for k=1,2,...,12.
    # These nodes grow quadratically, aligning with the asymptotic zero distribution
    # of Laguerre polynomials and potentially confining B more effectively.
    # We can scale the entire set of nodes by a factor s > 0.
    # The score B/(2*pi) is a function of s. We search for the s that minimizes it.
    
    # Base nodes (k*pi)^2
    k = np.arange(1, 13)
    base_nodes = (k * pi) ** 2
    
    # The scale factor s multiplies all nodes.
    # Since the polynomial g(x) is defined in terms of x, scaling x effectively
    # scales the zeros of the quotient q(x) in a non-trivial way, but for a
    # fixed set of relative positions, the "shape" of q is fixed.
    # Actually, the task is to find z_i. The evaluator constructs g from z_i.
    # If we scale z_i by s, the resulting B will scale by s as well?
    # Let's check: if we replace x by s*x in the problem, the nodes z_i become s*z_i.
    # The polynomial g_s(x) = g(x/s) satisfies the same conditions at s*z_i.
    # Then q_s(x) = g_s(x) / (x * prod (x - s*z_i)^2).
    # The roots of q_s are s times the roots of q.
    # So B_s = s * B_1.
    # To minimize B_s, we want s as small as possible?
    # But z_i must be > 0. There is no lower bound on z_i other than > 0.
    # Wait, the task says "z_i > 0". If we scale to 0, z_i -> 0.
    # However, the system of equations for coefficients c_k might become singular
    # or ill-conditioned if z_i are very small.
    # Also, the "singularity" check is for the linear system.
    # If we scale nodes by s, the matrix elements scale differently.
    # Let's just try to optimize the scale factor s.
    
    # The function to minimize is the score B/(2*pi).
    # We cannot evaluate the full Sympy construction inside minimize_scalar quickly.
    # We need a fast surrogate or we assume the scaling property.
    # If B scales linearly with s, then the optimal s is the smallest valid s.
    # What is the smallest valid s?
    # The nodes must be distinct and positive.
    # The linear system must be non-singular.
    # For very small s, the nodes are close to 0.
    # L_d^(alpha)(0) is finite.
    # g(0)=0, g'(0)=1.
    # g(z_i)=0, g'(z_i)=0.
    # If z_i -> 0, the conditions g(z_i)=0 and g'(z_i)=0 become conditions at 0.
    # g(0)=0 and g'(0)=1 are already given.
    # g(z_i)=0 for z_i near 0 implies g(x) ~ c x^2? But g'(0)=1 means g(x) ~ x.
    # So g(x) cannot have a zero of order 2 at 0.
    # Thus, for small z_i, the system is likely singular or ill-conditioned.
    
    # Let's look at the Bessel node strategy again.
    # The previous run found that Bessel nodes (k*pi)^2 gave a score of 9.882?
    # No, the cell finding said "Measured: 1 -> 9.882". This might mean the score was 9.882?
    # Or B was 9.882?
    # The current best is 0.4789.
    # If Bessel nodes give a much higher B, they are worse.
    # Wait, the inspiration 2 (score 0.4305) used "arithmetic spacing" and improved to Laguerre.
    # The current program uses Laguerre roots and scores 0.4789.
    # The cell finding "Using Bessel zeros... Measured: 1 -> 9.882" is ambiguous.
    # If it means the score jumped to 9.882, that's an improvement (lower is better? No, 9.882 > 0.4789).
    # "lower is better". So 0.4789 is better than 9.882.
    # So Bessel nodes are likely worse.
    
    # What if we scale the Laguerre roots?
    # Laguerre roots L_12^(-1/2) are in [0, ~30].
    # If we scale them by s, B scales by s.
    # To minimize B, we want s small.
    # But we must ensure the system is not singular.
    # Let's try to find the minimum s such that the system is well-conditioned.
    # We can estimate the condition number of the Vandermonde-like matrix.
    
    # The matrix A is built from L_2k(z_i) and derivatives.
    # For small z, L_2k(z) ~ L_2k(0) + z L'_2k(0).
    # The rows of the matrix for conditions at z_i and z_j will be similar if z_i ~ z_j.
    # The Laguerre roots are already spread out.
    # If we scale them down, they get closer to each other relative to their distance from 0?
    # No, they stay spread out but closer to 0.
    # The condition number might increase.
    
    # Let's try a grid search for s in [0.1, 1.0] on the Laguerre roots.
    # We can't run the full Sympy check here.
    # But we can assume that if the Laguerre roots are valid, scaled versions might be too,
    # until they become too clustered.
    
    # Let's try s = 0.5.
    # Or better, let's look at the "Chebyshev" attempt which scored 0.4540.
    # Chebyshev nodes on [0.1, 15.0].
    # Laguerre roots are roughly in [0.1, 25].
    # Chebyshev concentrates nodes near the ends.
    # Laguerre roots are distributed according to the weight.
    
    # What if we use the roots of L_12^(-1/2) but scale them?
    # Let's guess that the optimal scale is not 1.
    # The score is B/2pi.
    # If B scales with s, then score scales with s.
    # We want to minimize s.
    # Let's pick a small s, e.g., 0.1.
    # Risk: singularity.
    
    # Alternative: The nodes don't have to be Laguerre roots.
    # The problem is to minimize B.
    # B is the largest root of q.
    # q(x) = g(x) / (x prod (x-z_i)^2).
    # g is a linear combination of L_2k.
    # This is a complex optimization problem.
    
    # Given the constraint and the previous failure of Bessel nodes,
    # and the success of Laguerre nodes (0.4789) over Chebyshev (0.4540 is better? No 0.4540 < 0.4789).
    # Wait, 0.4540 is better than 0.4789.
    # So the Chebyshev node attempt (p27) scored 0.4540.
    # The current program (Laguerre) scores 0.4789.
    # So Chebyshev was better!
    # The previous response said "current score of 0.4789 is already higher than Chebyshev (0.4540)".
    # "higher" in score means worse.
    # So Chebyshev (0.4540) is better than Laguerre (0.4789).
    # We should revert to Chebyshev or improve upon it.
    
    # The Chebyshev nodes were on [0.1, 15.0].
    # Let's try to optimize the interval [L, R] for Chebyshev nodes.
    # Or simply use Chebyshev nodes on [0, 15] or similar.
    # The previous Chebyshev implementation:
    # L, R = 0.1, 15.0
    
    # Let's try to refine the Chebyshev nodes.
    # Maybe a different interval?
    # The roots of q depend on the spacing.
    # Let's try [0.1, 10.0] or [0.1, 20.0].
    
    # Since I can't evaluate, I will return the Chebyshev nodes from the known better run (0.4540).
    # And maybe tweak it slightly to ensure novelty?
    # Or just use it.
    
    n = 12
    L, R = 0.1, 15.0
    k = np.arange(n)
    nodes = 0.5 * (L + R) + 0.5 * (R - L) * np.cos(np.pi * (2 * k + 1) / (2 * n))
    nodes = sorted([float(z) for z in nodes])
    return nodes
