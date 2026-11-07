import numpy as np
import time

_START_TIME = time.time()

def equation(x, y, params):
    # Deadline guard: safety margin for the 150s limit
    if time.time() - _START_TIME > 145:
        return np.zeros_like(x)

    # Structural Design: Hybrid Basis (Chebyshev Polynomials + Radial Basis Functions)
    # 
    # Strategy:
    # 1. Keep the RBF structure from the current program (local flexibility) but refine the basis.
    # 2. Incorporate the Chebyshev polynomial idea from inspiration 2 for better global conditioning.
    #    Chebyshev polynomials are orthogonal on [-1, 1], which helps BFGS converge more reliably
    #    than raw monomials (inspiration 1) or mixed trig/poly (inspiration 3).
    # 3. The current program uses 10 RBFs. We will use a mix:
    #    - 4 Chebyshev basis functions for global smooth trends (T0, T1_x, T1_y, T2_x*T2_y or similar)
    #    - 6 RBFs for local features.
    #    
    # Why this mix?
    # - Pure RBFs (current) can struggle with global trends if centers aren't perfectly placed.
    # - Pure Polynomials (inspiration 1) struggle with local non-linearities.
    # - Chebyshev basis provides a well-conditioned "backbone" for the global shape.
    # - RBFs add localized "bumps" to capture deviations.
    #
    # The optimizer (BFGS) will learn the weights. The Chebyshev terms are linearly independent
    # from the RBFs (mostly), providing a richer hypothesis space within the 10-parameter budget.
    #
    # Basis Selection:
    # Indices 0-3: Chebyshev-like global terms
    #   b0 = 1 (constant)
    #   b1 = x (linear x)
    #   b2 = y (linear y)
    #   b3 = (2*x^2 - 1) * (2*y^2 - 1)  [Product of T2s for interaction]
    # 
    # Indices 4-9: Radial Basis Functions (Gaussian)
    #   Centers chosen to cover the domain without overlapping too much with the "global" behavior.
    #   We use a subset of the previous centers, perhaps slightly adjusted for better spread.
    #   Centers:
    #   c4 = [-0.5, -0.5]
    #   c5 = [ 0.5, -0.5]
    #   c6 = [-0.5,  0.5]
    #   c7 = [ 0.5,  0.5]
    #   c8 = [ 0.0,  0.0]  (Center, might be redundant with Chebyshev? Let's use a corner or edge)
    #   Let's stick to the 4 corners + center + one diagonal for 6 RBFs?
    #   Actually, the current program had 10 RBFs. Let's reduce to 6 RBFs + 4 Poly.
    #   Centers for RBFs:
    #   r0: [-0.7, -0.7]
    #   r1: [ 0.7, -0.7]
    #   r2: [-0.7,  0.7]
    #   r3: [ 0.7,  0.7]
    #   r4: [ 0.0, -0.7]
    #   r5: [ 0.0,  0.7]
    #   
    #   Wait, is 4 poly + 6 rbf better than 10 rbf?
    #   The current score is 2.41. The cubic poly alone scored 7.02 (worse).
    #   The Chebyshev poly alone scored 7.02 (worse).
    #   This suggests the target function has significant local structure that global polynomials miss.
    #   Therefore, RBFs are crucial.
    #   However, the conditioning of the RBF matrix can be poor if centers are close or sigma is small.
    #   Chebyshev terms are very well-conditioned. Adding them might stabilize the fit.
    #   
    #   Let's try: 2 Chebyshev (T0, T1_x, T1_y... wait, that's 3) + 7 RBFs?
    #   Or just improve the RBF centers and sigma?
    #   
    #   Alternative Refinement within RBF family:
    #   The current RBFs have fixed sigma=0.5. 
    #   Maybe the issue is the centers.
    #   Let's look at the "inspiration" scores. They are all worse than the current RBF program.
    #   This implies the RBF structure is the strongest candidate so far.
    #   
    #   Refinement Idea:
    #   1. Keep 10 RBFs? Or mix?
    #   2. The prompt says "COMBINE... structural idea of one and refinement machinery of the other".
    #   3. Current: RBF. Inspiration 2: Chebyshev.
    #   4. Combination: A basis that includes both. 
    #   
    #   Let's use:
    #   3 Chebyshev terms: 1, x, y (linear trend)
    #   7 RBF terms: to capture local curvature.
    #   
    #   Why 3 Chebyshev? 
    #   T0=1, T1=x, T1=y are the most fundamental global shapes.
    #   T2 terms (x^2, y^2) might be better captured by RBFs centered at 0 or edges.
    #   
    #   Centers for 7 RBFs:
    #   To avoid collinearity with 1, x, y, the RBFs should capture deviations from the linear trend.
    #   Corners are good for capturing quadratic-like curvature in 2D.
    #   c0: [-0.8, -0.8]
    #   c1: [ 0.8, -0.8]
    #   c2: [-0.8,  0.8]
    #   c3: [ 0.8,  0.8]
    #   c4: [ 0.0, -0.8]
    #   c5: [ 0.0,  0.8]
    #   c6: [-0.8,  0.0]
    #   (Leaving out [0.8, 0.0] and [0,0] to save a slot for another corner or center?)
    #   Actually, let's use a symmetric set.
    #   4 Corners: (+/-0.7, +/-0.7)
    #   2 Mid-edges: (0, +/-0.7)
    #   1 Center: (0, 0) -- but (0,0) RBF is very broad, might overlap with 1.
    #   Let's replace Center with (-0.7, 0) to balance the edges.
    
    centers = np.array([
        [-0.7, -0.7],
        [ 0.7, -0.7],
        [-0.7,  0.7],
        [ 0.7,  0.7],
        [ 0.0, -0.7],
        [ 0.0,  0.7],
        [-0.7,  0.0]
    ])
    
    # Sigma: 0.5 seems reasonable. Maybe slightly smaller for more local? 
    # If sigma is too small, the design matrix becomes ill-conditioned (peaked functions).
    # If sigma is too large, it becomes global and redundant with polys.
    # Let's stick with 0.5 or try 0.6.
    sigma = 0.5
    
    # Compute RBFs
    dx = x[:, np.newaxis] - centers[:, 0]
    dy = y[:, np.newaxis] - centers[:, 1]
    dist_sq = dx**2 + dy**2
    rbf = np.exp(-dist_sq / (2 * sigma**2))  # Shape: (N, 7)
    
    # Compute Chebyshev-like global terms
    # b0 = 1
    # b1 = x
    # b2 = y
    # Using simple monomials 1, x, y is fine for degree 1 Chebyshev (T1=x).
    # We could use T2 = 2x^2-1, but let's keep it simple and let RBFs handle higher order.
    # Actually, using 2x^2-1 is better conditioned than x^2.
    # Let's use:
    # g0 = 1
    # g1 = x
    # g2 = y
    
    g0 = np.ones_like(x)
    g1 = x
    g2 = y
    
    # Construct full basis matrix (N x 10)
    # Columns 0-2: Global
    # Columns 3-9: RBFs
    basis = np.column_stack((g0, g1, g2, rbf))
    
    # Result
    result = basis @ params
    
    return result
