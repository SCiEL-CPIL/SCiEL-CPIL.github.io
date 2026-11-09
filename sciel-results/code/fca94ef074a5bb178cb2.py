def penalty_vec(points, r):
    pts = points.reshape(N, 2)
    # Pairwise distances
    # diff shape (N, N, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Mask i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    dist_pair = dist[mask]
    viol_pair = np.maximum(0, 2 * r - dist_pair)
    pen_pair = np.sum(viol_pair ** 2)
    
    # Boundary
    dist_bound = np.linalg.norm(pts, axis=1)
    viol_bound = np.maximum(0, dist_bound - (1 - r))
    pen_bound = np.sum(viol_bound ** 2)
    
    return pen_pair + pen_bound
