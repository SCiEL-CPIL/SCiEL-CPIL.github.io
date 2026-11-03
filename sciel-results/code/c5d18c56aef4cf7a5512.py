<<<<<<< SEARCH
def _negative_soft_radius_grad(pts, beta=50.0):
    """Gradient of negative soft radius."""
    n = len(pts)
    dists_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    half_dists = dist_matrix / 2.0
    
    all_vals = np.concatenate([dists_boundary, half_dists.ravel()])
    m = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - m))
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(pts)
    
    # Boundary part: d/dx_i of (1 - |x_i|) = -x_i/|x_i|
    for i in range(n):
        r = np.hypot(pts[i, 0], pts[i, 1])
        if r > 1e-12:
            grad[i] += weights[i] * (-pts[i] / r)
        # Note: we want gradient of -soft_min, so negate
        # Actually soft_min ≈ m - log(sum(exp(-beta*(arr-m))))/beta
        # d(soft_min)/d(arr_k) = -exp(-beta*(arr_k-m)) / sum(exp(-beta*(arr-m))) = -weights[k]
        # d(-soft_min)/d(arr_k) = weights[k]
        # d(arr_k)/d(pts) gives us the gradient contribution
    
    # Pairwise part: for pair (i,j), half_dist = |x_i - x_j|/2
    # d(half_dist)/d(x_i) = (x_i - x_j) / (2 * |x_i - x_j|)
    for i in range(n):
        for j in range(i+1, n):
            d = dist_matrix[i, j]
            if d > 1e-12:
                w_ij = weights[n + i*n + j] + weights[n + j*n + i]
                direction = (pts[i] - pts[j]) / d
                grad[i] += w_ij * 0.5 * direction
                grad[j] += w_ij * (-0.5 * direction)
    
    # The full gradient of -soft_min w.r.t. pts is sum_k weights[k] * d(arr_k)/d(pts)
    # But we computed contributions with signs already... let me redo carefully.
    
    # soft_min(arr) = m - (1/beta)*log(sum(exp(-beta*(arr_k - m))))
    # d(soft_min)/d(arr_k) = -exp(-beta*(arr_k-m)) / sum(exp(-beta*(arr-m))) = -w_k
    # d(-soft_min)/d(arr_k) = +w_k
    # So grad(-soft_min) = sum_k w_k * d(arr_k)/d(pts)
    
    # Boundary: arr_i = 1 - |p_i|, d(arr_i)/d(p_i) = -p_i/|p_i|
    # Pair (i,j): arr = |p_i-p_j|/2, d(arr)/d(p_i) = (p_i-p_j)/(2|p_i-p_j|), d(arr)/d(p_j) = (p_j-p_i)/(2|p_i-p_j|)
    
    grad = np.zeros_like(pts)
    for i in range(n):
        r = np.hypot(pts[i, 0], pts[i, 1])
        if r > 1e-12:
            grad[i] += weights[i] * (-pts[i] / r)
    
    for i in range(n):
        for j in range(i+1, n):
            d = dist_matrix[i, j]
            if d > 1e-12:
                w_ij = weights[n + i*n + j] + weights[n + j*n + i]
                direction = (pts[i] - pts[j]) / d
                grad[i] += w_ij * 0.5 * direction
                grad[j] += w_ij * 0.5 * (-direction)
    
    return grad
=======
def _negative_soft_radius_grad(pts, beta=50.0):
    """Gradient of negative soft radius (vectorized)."""
    n = len(pts)
    dists_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    half_dists = dist_matrix / 2.0
    
    all_vals = np.concatenate([dists_boundary, half_dists.ravel()])
    m = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - m))
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(pts)
    
    # Boundary contribution
    r = np.hypot(pts[:, 0], pts[:, 1])
    mask_b = r > 1e-12
    grad[mask_b] += weights[mask_b][:, np.newaxis] * (-pts[mask_b] / r[mask_b][:, np.newaxis])
    
    # Pairwise contribution: vectorized
    w_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            w_matrix[i, j] = weights[n + i*n + j]
            w_matrix[j, i] = weights[n + j*n + i]
    
    # For each pair (i,j), contribution to grad[i] = w_ij * 0.5 * (p_i - p_j) / |p_i - p_j|
    # Vectorize: grad[i] = sum_j w_matrix[i,j] * 0.5 * (p_i - p_j) / d_ij
    d_safe = np.where(dist_matrix > 1e-12, dist_matrix, 1.0)
    direction_matrix = diff / d_safe[:, :, np.newaxis]
    direction_matrix = np.where(dist_matrix[:, :, np.newaxis] > 1e-12, direction_matrix, 0.0)
    
    # grad[i] = sum_j w_matrix[i,j] * 0.5 * direction_matrix[i,j]
    grad += 0.5 * np.sum(w_matrix[:, :, np.newaxis] * direction_matrix, axis=1)
    
    return grad
>>>>>>> REPLACE

