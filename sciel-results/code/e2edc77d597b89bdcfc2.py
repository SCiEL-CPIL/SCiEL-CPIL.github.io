import numpy as np

def erdos_szekeres_problem7() -> np.ndarray:
    n = 33
    # Generate points on a cubic curve y = x^3
    # This curve has an inflection point, so not all points are in convex position.
    # The convex hull will be relatively small, and many points will be interior.
    
    # x in [-1, 1]
    x = np.linspace(-1, 1, n)
    y = x**3
    
    # Map to [0.1, 0.9] to stay safely inside [0,1]
    # x -> 0.5 + 0.4 * x
    # y -> 0.5 + 0.4 * y (since y is in [-1, 1])
    
    points = np.zeros((n, 2))
    points[:, 0] = 0.5 + 0.4 * x
    points[:, 1] = 0.5 + 0.4 * y
    
    # Add small perturbations to avoid collinearity and exact duplicates
    # The slope of y=x^3 is 3x^2. At x=0, slope is 0.
    # Perturbations should be small enough not to change the "convexity" logic drastically
    # but large enough to break exact collinearity.
    # 1e-4 is usually safe.
    
    rng = np.random.default_rng(123)
    perturbation = rng.uniform(-1e-4, 1e-4, size=points.shape)
    points += perturbation
    
    # Ensure points are within [0, 1]
    # The base points are in [0.1, 0.9]. Perturbation is 1e-4.
    # So points are in [0.0999, 0.9001]. Safe.
    
    return points
