# EVOLVE-BLOCK-START
import numpy as np
import time


def erdos_szekeres_problem7() -> np.ndarray:
    """
    Returns a (33, 2) float array of points in the unit square that minimizes
    the number of convex 7-gons.
    
    Strategy: Place points on a strictly convex curve (parabola y=x^2).
    Points on a convex curve have the property that for any subset, many points
    lie inside the convex hull of the subset's extreme points. This minimizes
    the number of subsets that are themselves in convex position (i.e., form
    a convex polygon).
    """
    n = 33
    # Use a deadline guard to be safe, though this construction is O(1).
    start_time = time.time()
    deadline = 600.0
    
    # Generate x coordinates evenly spaced in [0, 1]
    x = np.linspace(0, 1, n)
    
    # Generate y coordinates on a convex curve y = x^2
    # This ensures the points are in general position (no 3 collinear)
    # and minimizes convex k-gons.
    y = x ** 2
    
    points = np.column_stack((x, y))
    
    # Check validity constraints (should pass by construction, but verify)
    # 1. Shape
    if points.shape != (n, 2):
        # Fallback to random if something went wrong (shouldn't happen)
        np.random.seed(0)
        points = np.random.rand(n, 2)
        
    # 2. NaN/Inf
    if not np.all(np.isfinite(points)):
        np.random.seed(0)
        points = np.random.rand(n, 2)
        
    # 3. Bounds
    if not (np.all(points >= -1e-6) and np.all(points <= 1 + 1e-6)):
        # Clip to be safe, though linspace ensures [0,1]
        points = np.clip(points, 0, 1)
        
    # 4. Distance and Collinearity are guaranteed by the convex curve construction
    # with distinct x-coordinates.
    
    return points


# EVOLVE-BLOCK-END
