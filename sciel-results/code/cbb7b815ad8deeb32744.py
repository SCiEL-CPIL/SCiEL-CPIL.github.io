# EVOLVE-BLOCK-START
import numpy as np
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points.

    Returns:
        points: np.ndarray of shape (num_points,11)
    """
    d = 11
    deadline = time.time() + 600
    
    # Generate candidate vectors: all vectors with entries in {-1, 0, 1}
    # with exactly 2 non-zero entries (norm^2 = 2)
    candidates = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Also add single-coordinate vectors (norm^2 = 1) as fallback
    for i in range(d):
        for si in [-1, 1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = si
            candidates.append(v)
    
    # Sort candidates by norm (prefer smaller norms for easier packing)
    norms = [np.dot(v, v) for v in candidates]
    order = np.argsort(norms)
    candidates = [candidates[i] for i in order]
    norms = [norms[i] for i in order]
    
    selected = []
    selected_norms = []
    
    for idx, v in enumerate(candidates):
        if time.time() > deadline - 60:
            break
        
        v_norm2 = norms[idx]
        
        # Check against all selected points
        ok = True
        for s_idx, s in enumerate(selected):
            diff = v - s
            dist2 = np.dot(diff, diff)
            if dist2 < v_norm2 or dist2 < selected_norms[s_idx]:
                ok = False
                break
        
        if ok:
            selected.append(v)
            selected_norms.append(v_norm2)
    
    if not selected:
        # Fallback: at least 2 points
        selected = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
    
    points = np.array(selected, dtype=np.int64)
    points = np.around(points).astype(np.int64)
    
    return points


# EVOLVE-BLOCK-END
