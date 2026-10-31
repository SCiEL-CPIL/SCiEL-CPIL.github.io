def min_triangle_area(points):
    n = len(points)
    if n < 3:
        return 0.0
    # Vectorized area calculation
    # Using broadcasting to compute all pairwise differences is O(n^2), then selecting triples is complex.
    # For N=11, a simple loop is actually quite fast in Python (165 iterations).
    # The bottleneck might be elsewhere.
    # However, let's use a slightly optimized loop or vectorized approach.
    
    # Let's try a vectorized approach using indices
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    # Cross product
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas) / AREA
