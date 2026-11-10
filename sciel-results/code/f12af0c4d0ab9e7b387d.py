import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    n_vecs = 1 << d
    
    # Precompute hamming weights
    weights = np.array([bin(i).count('1') for i in range(n_vecs)], dtype=np.uint8)
    
    selected = [0]
    
    # To improve packing, we can try to add vectors in an order that 
    # maximizes the minimum distance to the current set? 
    # No, just standard order is fine for a baseline.
    # But we can do a simple heuristic: 
    # Iterate through all vectors.
    
    for v in range(1, n_vecs):
        # Check if v is at distance >= 3 from all selected
        # dist(v, u) = weights[v ^ u]
        # We can vectorize this check against the selected list
        # But selected list grows.
        
        # Simple loop
        is_valid = True
        for u in selected:
            if weights[v ^ u] < 3:
                is_valid = False
                break
        if is_valid:
            selected.append(v)
            
    k = len(selected)
    points = np.zeros((k, d), dtype=np.int64)
    
    # Map binary to {-1, 1}
    # v's j-th bit (0-indexed) -> points[i, j]
    # 0 -> -1, 1 -> 1
    # points[i, j] = 2 * bit - 1
    
    for i, v in enumerate(selected):
        # Extract bits
        # Use bit operations
        for j in range(d):
            if (v >> j) & 1:
                points[i, j] = 1
            else:
                points[i, j] = -1
                
    return points
