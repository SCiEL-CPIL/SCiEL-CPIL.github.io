import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the target function using a Multi-Scale Gaussian RBF basis.
    
    The model uses 10 parameters:
    - 5 weights for wide Gaussians (sigma=1.0) centered at -2, -1, 0, 1, 2
    - 5 weights for narrow Gaussians (sigma=0.4) centered at -1.5, -0.5, 0.5, 1.5, and 0 (shifted to avoid overlap issues? No, let's space them out)
    
    Actually, let's use 10 centers for wide and 10 for narrow? No, we only have 10 params.
    
    Strategy:
    Split the 10 params into two groups of 5.
    Group 1: Wide Gaussians (sigma=1.2) at centers: -2, -1, 0, 1, 2
    Group 2: Narrow Gaussians (sigma=0.5) at centers: -1.5, -0.5, 0.5, 1.5, 0 (Wait, 0 is already used. Let's use -1.5, -0.5, 0.5, 1.5, and maybe 0 is covered by wide? No, let's put narrow at -1.5, -0.5, 0.5, 1.5, and 2? No, 2 is edge.
    Let's use centers for narrow: -1.5, -0.5, 0.5, 1.5, and 0.
    Wait, if I use 0 for both, it's fine.
    
    Let's refine centers:
    Wide centers: -2, -1, 0, 1, 2
    Narrow centers: -1.5, -0.5, 0.5, 1.5, 0 (This puts a narrow bump at 0, which is also a wide bump center. This is okay, it adds local detail at 0.)
    
    Alternatively, use all 10 params for 10 Gaussians with a single width, but choose width to be adaptive? No, fixed width is safer for BFGS.
    
    Let's try a different structural change: **B-spline Basis**.
    B-splines are local and smooth.
    We can construct a B-spline basis of order 4 (cubic) with 10 parameters.
    This is equivalent to a piecewise cubic polynomial with 7 knots?
    With 10 parameters, we can have 10 basis functions.
    B-splines are non-negative and sum to 1 (partition of unity). This is great for stability.
    However, B-splines are bounded. If the target function grows large (like $e^x$), we need large weights. That's fine.
    
    Let's implement a simple **Cubic B-spline** basis.
    We need to define the knots.
    Domain [-2, 2].
    Let's use uniform knots.
    Number of basis functions = 10.
    This implies 10 control points.
    
    Actually, implementing B-splines from scratch in `equation` is prone to errors.
    The Gaussian RBF is simpler and more robust to implement correctly.
    
    Let's stick to **Multi-Scale Gaussian RBF**.
    
    Centers Wide: [-2, -1, 0, 1, 2]
    Centers Narrow: [-1.5, -0.5, 0.5, 1.5, 0]
    
    Wait, if the target is a polynomial, RBFs might be worse than Chebyshev.
    But Chebyshev scored 190. This suggests the target is NOT well-approximated by a degree 9 polynomial.
    This strongly points to a function with a **kink**, **discontinuity**, or **sharp peak**.
    RBFs are excellent for modeling sharp peaks and local features.
    
    Let's use 10 Gaussians with a moderate width, say sigma=0.6, centered at:
    -2, -1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6
    (Leaving out 2? Or shifting?)
    Let's use 10 points from -2 to 2:
    np.linspace(-2, 2, 10) -> -2, -1.55, -1.11, -0.66, -0.22, 0.22, 0.66, 1.11, 1.55, 2.
    This covers the domain well.
    Width sigma = 0.5.
    
    This allows the model to fit local variations.
    """
    x = np.asarray(x)
    
    # Define 10 centers
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Define width
    sigma = 0.5
    
    # Compute Gaussians
    # x is shape (N,), centers is shape (10,)
    # We want shape (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    gaussians = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # params is shape (10,)
    # result = gaussians @ params
    result = np.dot(gaussians, params)
    
    return result
