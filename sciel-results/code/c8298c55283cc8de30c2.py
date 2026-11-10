import numpy as np
import time

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # Generate all vectors with exactly two non-zero entries, each +/- 1
    # These have squared norm 2.
    # Min squared distance between any two such distinct vectors is 2.
    # Max squared norm is 2.
    # Condition: min_dist >= max_norm => 2 >= 2. Valid.
    # Count: C(11, 2) * 4 = 55 * 4 = 220.
    
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    pts_array = np.array(points, dtype=np.int64)
    
    # Optional: Verify to be safe, though analysis is solid.
    # Verification is O(k^2) = 220^2 = 48400 ops. Very fast.
    n_pts = len(pts_array)
    if n_pts > 0:
        norms = np.sum(pts_array ** 2, axis=1)
        max_norm = int(norms.max())
        
        min_dist = float('inf')
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                diff = pts_array[i] - pts_array[j]
                dist_sq = int(np.sum(diff ** 2))
                if dist_sq < min_dist:
                    min_dist = dist_sq
        
        if min_dist < max_norm:
            # Fallback to 22 unit vectors if somehow invalid
            pts_array = np.array([np.eye(d, dtype=np.int64)[i] * s for i in range(d) for s in [1, -1]])
            
    return np.around(pts_array).astype(np.int64)
