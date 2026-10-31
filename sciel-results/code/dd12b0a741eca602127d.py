def _min_area_vec(pts):
    # pts: (N, 2)
    # N=11
    # Generate all combinations of indices
    idx = list(combinations(range(N), 3))
    # pts[idx] is not direct.
    # pts[0], pts[1], pts[2]
    # We can stack them.
    p1 = np.array([pts[i] for i,j,k in idx])
    p2 = np.array([pts[j] for i,j,k in idx])
    p3 = np.array([pts[k] for i,j,k in idx])
    
    # Vectors
    v1 = p2 - p1
    v2 = p3 - p1
    # Cross product z-component
    # v1_x * v2_y - v1_y * v2_x
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)
