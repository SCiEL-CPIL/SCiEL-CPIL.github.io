def _features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a (n, 10) feature matrix covering common smooth functions.
    
    Weights are chosen to cover:
    - Constant, linear, quadratic terms (full polynomial up to degree 2)
    - Key cubic terms for better approximation
    - Radial and interaction terms for non-polynomial structure
    """
    x2 = x * x
    y2 = y * y
    xy = x * y
    # r2 = x2 + y2 # Removed redundant feature

    # Constant
    f0 = np.ones_like(x)
    # Linear
    f1 = x
    f2 = y
    # Quadratic
    f3 = x2
    f4 = y2
    f5 = xy
    # Replace redundant f6 (x2+y2) with x2*y2 (degree 4 interaction)
    f6 = x2 * y2
    # Cubic terms
    f7 = x * x2        # x^3
    f8 = y * y2        # y^3
    # Mixed cubic: x^2*y + x*y^2 = xy(x+y)
    f9 = xy * (x + y)

    F = np.stack([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9], axis=1)
    return F
