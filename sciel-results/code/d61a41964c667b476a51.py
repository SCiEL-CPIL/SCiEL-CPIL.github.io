import numpy as np

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses all vectors of Hamming weight 2.
    
    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    d = 11
    # Generate all pairs (i, j) with 0 <= i < j < d
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            v = np.zeros(d, dtype=np.int64)
            v[i] = 1
            v[j] = 1
            points.append(v)
    
    if not points:
        # Fallback, though d=11 ensures points exist
        return np.array([np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)])
        
    arr = np.array(points)
    return arr
