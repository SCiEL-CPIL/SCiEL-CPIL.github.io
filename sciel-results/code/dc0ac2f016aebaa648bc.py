import numpy as np
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses the vertices of the cross-polytope (standard basis vectors and their negatives).
    
    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    d = 11
    # Create standard basis vectors
    basis = np.eye(d, dtype=np.int64)
    
    # The set of points is {e_1, ..., e_d, -e_1, ..., -e_d}
    # This gives 2*d points.
    points = np.vstack([basis, -basis])
    
    return points
