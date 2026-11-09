def lloyd_step(points):
    dots = points @ points.T
    nearest_indices = np.argmax(dots, axis=1)
    
    # Create assignment matrix
    # M[i, j] = 1 if nearest_indices[j] == i
    M = np.zeros((len(points), len(points)), dtype=float)
    indices = np.arange(len(points))
    M[nearest_indices, indices] = 1.0
    
    counts = np.sum(M, axis=1)
    sums = M.T @ points
    
    # Avoid division by zero
    counts_safe = np.where(counts == 0, 1.0, counts)
    new_points = sums / counts_safe[:, np.newaxis]
    
    # Normalize
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return new_points / norms
