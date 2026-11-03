def _gradient_energy_fast(points):
    """Vectorized gradient calculation."""
    # diff[i,j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # dist_sq[i,i] is 0. We need 1/dist^2.
    # Mask diagonal to avoid division by zero
    dist_sq_safe = dist_sq.copy()
    np.fill_diagonal(dist_sq_safe, 1.0) # Avoid div by 0
    
    # term[i,j] = diff[i,j] / dist_sq[i,j]
    # grad[i] = - sum_j term[i,j]  (since grad[i] -= diff[i,j]/d^2)
    # Wait, the formula is:
    # grad[i] -= (x_i - x_j) / d_ij^2
    # grad[j] += (x_i - x_j) / d_ij^2
    # So grad[i] = - sum_{j != i} (x_i - x_j) / d_ij^2
    
    coeff = 1.0 / dist_sq_safe
    coeff_diag = np.eye(N)
    coeff = coeff * (1.0 - coeff_diag) # Zero out diagonal
    
    # grad = - sum_j (points[i] - points[j]) * coeff[i,j]
    # = - points[i] * sum_j coeff[i,j] + sum_j points[j] * coeff[i,j]
    
    sum_coeff = np.sum(coeff, axis=1) # Shape (N,)
    # sum_j points[j] * coeff[i,j]
    # This is a matrix multiplication: coeff @ points
    # coeff is (N,N), points is (N,3) -> result (N,3)
    sum_points_weighted = coeff @ points
    
    grad = -points * sum_coeff[:, np.newaxis] + sum_points_weighted
    
    return grad
