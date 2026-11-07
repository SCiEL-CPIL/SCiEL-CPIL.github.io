import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Sparse Adaptive Basis Pursuit.
    
    Instead of relying on a fixed small set of parameters (like RBF centers or 
    polynomial coefficients) which forces the optimizer into a potentially 
    poor local minimum or underfitting regime, we treat the problem as a 
    sparse linear regression over a large library of candidate features.
    
    The 'params' array is ignored for the structural definition but used to 
    seed the initial weights if needed, though we primarily rely on internal 
    optimization to find the best sparse combination of basis functions.
    
    Basis Library:
    - Constant
    - Linear terms (x, y, z)
    - Quadratic terms (x^2, y^2, z^2, xy, xz, yz)
    - Cubic terms (x^3, y^3, z^3, x^2y, x^2z, y^2x, y^2z, z^2x, z^2y, xyz)
    - Trigonometric/Exponential terms (sin, cos, exp) to capture oscillatory or 
      saturating behaviors.
    
    We use a simple, fast L1-regularized coordinate descent (Lasso-like) or 
    simply least squares on a pre-selected subset if time permits, but given 
    the 120s limit and 200 points, we can afford a moderate number of iterations 
    of a soft-thresholding algorithm or a greedy forward selection.
    
    However, scipy.optimize.minimize is called BY THE EVALUATOR on our function.
    Wait, the contract says: "The evaluator fits params... minimizing the training MSE."
    This means our function is the MODEL. The evaluator calls equation(x, y, z, params) 
    with different params to compute residuals.
    
    CRITICAL REALIZATION:
    If I implement internal optimization inside `equation`, it will run EVERY TIME 
    the evaluator calls `equation` during the BFGS iterations. This will be 
    catastrophically slow (BFGS might call it 50-100 times, and if each call 
    takes 10s, we timeout).
    
    Therefore, `equation` must be a FAST, CLOSED-FORM expression defined by `params`.
    
    So, how do we get the "Sparse Adaptive" benefit without internal optimization?
    We must choose a FIXED basis that is rich enough, but structured such that 
    BFGS can easily find the coefficients.
    
    The previous failure (score 253) was likely because the RBF centers were 
    non-convex to optimize.
    The inspirations (score ~7.4) used "Fixed-Basis Linear Regression with 
    Quadratic + Interaction Terms".
    
    Let's try a DIFFERENT structural change: 
    Instead of just polynomials, let's use a **Fourier-like basis** or 
    **Chebyshev polynomials** which are better conditioned on the domain [0.5, 2].
    Or, we can use a **Product of Sigmoid/Tanh functions** to create a 
    "Neural Network" with a FIXED topology where the parameters are the 
    weights and biases, but we fix the activation to be smooth and 
    well-conditioned.
    
    Actually, the most robust "structural" change that avoids non-convexity 
    (centers) while being more flexible than pure polynomials is a **Fixed 
    Radial Basis Function Network with FIXED centers and widths**, where 
    only the WEIGHTS are optimized by BFGS.
    
    If the centers and widths are fixed (e.g., on a grid), the problem becomes 
    a linear least squares problem (convex). BFGS will converge globally to 
    the best linear combination.
    
    Let's define a grid of centers in [0.5, 2]^3.
    With 10 parameters, we can't have 10 RBFs with fixed centers AND optimize 
    weights if we want to use all 10 params as weights.
    
    Let's use 10 fixed RBF centers.
    Centers: A 2x2x2 grid in the domain [0.5, 2].
    x_coords = [0.5, 2.0], y_coords = [0.5, 2.0], z_coords = [0.5, 2.0].
    This gives 8 centers. We can add 2 more at the center (1.25, 1.25, 1.25) 
    and maybe one more? Or just use 8 centers and 2 linear terms?
    
    Let's use 8 RBF centers (fixed) + 2 linear terms (x, y) or just 8 RBFs 
    and 2 constant/linear terms to make 10.
    
    Actually, a better distribution:
    Let's use 10 centers.
    We can place them on a sparse grid.
    
    Model: f(x,y,z) = sum_{i=1}^{10} w_i * exp(-||p - c_i||^2 / (2 * sigma^2))
    
    Here, c_i and sigma are FIXED constants. w_i are the 10 parameters.
    This is a linear model in w. BFGS will solve it perfectly (or close to it) 
    as long as the basis is not collinear.
    
    Why this is better than the current program:
    Current: Optimizes centers AND weights (non-convex).
    New: Optimizes ONLY weights (convex).
    
    Why this is better than the inspiration (polynomials):
    Polynomials might miss local structures. RBFs with fixed centers 
    distributed across the domain can capture local bumps/dips that 
    polynomials (especially low degree) cannot.
    
    Let's define the fixed centers.
    Domain: [0.5, 2.0].
    Let's use a 2x2x2 grid: 8 points.
    Plus 2 points at the center of the domain?
    Or a 3x3x3 grid is 27 points, too many for 10 params.
    
    Let's use 10 specific centers:
    1. (0.5, 0.5, 0.5)
    2. (0.5, 0.5, 2.0)
    3. (0.5, 2.0, 0.5)
    4. (0.5, 2.0, 2.0)
    5. (2.0, 0.5, 0.5)
    6. (2.0, 0.5, 2.0)
    7. (2.0, 2.0, 0.5)
    8. (2.0, 2.0, 2.0)
    9. (1.25, 1.25, 1.25)  # Center
    10. (1.25, 0.5, 1.25)   # Edge center?
    
    Actually, symmetry is good.
    Let's use 8 corners + 2 face centers?
    
    Let's stick to the 8 corners + 2 additional points to break symmetry 
    if needed, or just 8 corners + 1 constant + 1 linear?
    No, let's use 10 RBFs.
    
    Centers:
    c1 = (0.5, 0.5, 0.5)
    c2 = (0.5, 0.5, 2.0)
    c3 = (0.5, 2.0, 0.5)
    c4 = (0.5, 2.0, 2.0)
    c5 = (2.0, 0.5, 0.5)
    c6 = (2.0, 0.5, 2.0)
    c7 = (2.0, 2.0, 0.5)
    c8 = (2.0, 2.0, 2.0)
    c9 = (1.25, 1.25, 0.5)
    c10 = (1.25, 1.25, 2.0)
    
    Width (sigma): 
    The domain size is 1.5. A good sigma is around 0.5-0.8 to allow overlap 
    but maintain locality. Let's use sigma = 0.6.
    
    This model is linear in params. BFGS will converge fast.
    """
    try:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        p = np.asarray(params, dtype=np.float64)
        
        # Ensure p is 1D and length 10
        if p.shape != (10,):
            p = np.resize(p, 10)
            
        # Fixed Centers
        centers = np.array([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 2.0],
            [0.5, 2.0, 0.5],
            [0.5, 2.0, 2.0],
            [2.0, 0.5, 0.5],
            [2.0, 0.5, 2.0],
            [2.0, 2.0, 0.5],
            [2.0, 2.0, 2.0],
            [1.25, 1.25, 0.5],
            [1.25, 1.25, 2.0]
        ])
        
        # Fixed Width
        sigma = 0.6
        sigma_sq = sigma * sigma
        
        # Compute distances from each point (x,y,z) to each center
        # x, y, z are arrays of shape (N,)
        # centers are (10, 3)
        
        # We can vectorize this:
        # Points: (N, 3)
        points = np.column_stack((x, y, z))
        
        # Diff: (N, 1, 3) - (1, 10, 3) -> (N, 10, 3)
        diffs = points[:, np.newaxis, :] - centers[np.newaxis, :, :]
        
        # Squared distances: (N, 10)
        dist_sq = np.sum(diffs ** 2, axis=2)
        
        # RBF values: (N, 10)
        rbf_values = np.exp(-dist_sq / (2.0 * sigma_sq))
        
        # Linear combination: (N, 10) @ (10,) -> (N,)
        output = rbf_values @ p
        
        # Safety check
        if not np.all(np.isfinite(output)):
            return np.zeros_like(x)
            
        return output
        
    except Exception:
        try:
            return np.zeros(len(x))
        except:
            return np.array([0.0])
