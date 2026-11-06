<<<<<<< SEARCH
def _construct_single_start(pts_init, maxiter, penalty_weight):
    """Run L-BFGS-B from a single initial configuration."""
    def objective_and_grad(pts):
        pts = pts.reshape(-1, 2)
        norms = np.linalg.norm(pts, axis=1)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # We want to maximize r = min(min_dist/2, 1-max_norm)
        # Use a smooth approximation: minimize -log(min_dist/2) - log(1-max_norm) + penalty
        # Actually, let's use a direct approach: minimize the sum of violations
        
        # Target: all dists >= 2r, all norms <= 1-r
        # Let's minimize: sum_{i<j} max(0, 2r - dist_ij)^2 + sum_i max(0, r + norm_i - 1)^2
        # with r = min(min_dist/2, 1-max_norm)
        
        min_dist = np.min(dists)
        min_boundary = 1.0 - np.max(norms)
        r = min(min_dist / 2.0, min_boundary)
        
        if r <= 0:
            # Invalid configuration, return large penalty
            val = 1e6
            grad = np.zeros_like(pts)
            return val, grad
        
        # Penalty for pair violations
        pair_viol = np.maximum(0, 2*r - dists)
        pair_viol[np.diag_indices_from(pair_viol)] = 0
        
        # Penalty for boundary violations
        bound_viol = np.maximum(0, r + norms - 1.0)
        
        val = np.sum(pair_viol**2) + penalty_weight * np.sum(bound_viol**2)
        
        # Gradient computation
        grad = np.zeros_like(pts)
        
        # Pair gradient
        for i in range(N):
            for j in range(i+1, N):
                if pair_viol[i, j] > 0:
                    d = pts[i] - pts[j]
                    norm_d = np.linalg.norm(d)
                    if norm_d > 1e-12:
                        grad[i] += 4 * pair_viol[i, j] * d / norm_d
                        grad[j] -= 4 * pair_viol[i, j] * d / norm_d
        
        # Boundary gradient
        for i in range(N):
            if bound_viol[i] > 0:
                norm_i = norms[i]
                if norm_i > 1e-12:
                    grad[i] += 2 * penalty_weight * bound_viol[i] * pts[i] / norm_i
        
        return val, grad
    
    try:
        result = minimize(objective_and_grad, pts_init.reshape(-1), method='L-BFGS-B',
                          jac=True, options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
        pts_opt = result.x.reshape(-1, 2)
        return pts_opt, _score(pts_opt)
    except Exception:
        return pts_init, _score(pts_init)
=======
def _construct_single_start(pts_init, maxiter, penalty_weight):
    """Run L-BFGS-B from a single initial configuration."""
    # Safety check
    if pts_init.shape[0] != N:
        # Pad or truncate to N? Better to skip or raise.
        # For now, just return the input score to avoid crash
        return pts_init, _score(pts_init)

    def objective_and_grad(pts):
        pts = pts.reshape(-1, 2)
        norms = np.linalg.norm(pts, axis=1)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_dist = np.min(dists)
        min_boundary = 1.0 - np.max(norms)
        r = min(min_dist / 2.0, min_boundary)
        
        if r <= 0:
            val = 1e6
            grad = np.zeros_like(pts)
            return val, grad
        
        # Penalty for pair violations
        pair_viol = np.maximum(0, 2*r - dists)
        pair_viol[np.diag_indices_from(pair_viol)] = 0
        
        # Penalty for boundary violations
        bound_viol = np.maximum(0, r + norms - 1.0)
        
        val = np.sum(pair_viol**2) + penalty_weight * np.sum(bound_viol**2)
        
        # Vectorized Gradient computation
        grad = np.zeros_like(pts)
        
        # Pair gradient: grad_i = sum_{j != i} 4 * max(0, 2r - d_ij) * (p_i - p_j) / d_ij
        # We can compute this using matrix operations
        # factor_ij = 4 * pair_viol_ij / d_ij
        # factor_ii = 0
        
        # Avoid division by zero
        safe_dists = np.where(dists > 1e-12, dists, 1.0)
        factors = 4.0 * pair_viol / safe_dists
        np.fill_diagonal(factors, 0.0)
        
        # grad_i = sum_j factors_ij * (p_i - p_j)
        # grad_i = sum_j factors_ij * p_i - sum_j factors_ij * p_j
        # grad_i = p_i * sum_j factors_ij - sum_j (factors_ij * p_j)
        
        sum_factors = np.sum(factors, axis=1)
        # factors is (N, N). p is (N, 2).
        # sum_j (factors_ij * p_j) is matrix-vector product factors @ p
        grad_pairs = sum_factors[:, np.newaxis] * pts - (factors @ pts)
        
        # Boundary gradient
        # grad_i = 2 * penalty_weight * bound_viol_i * p_i / norm_i
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        bound_factors = np.where(bound_viol > 0, 2.0 * penalty_weight * bound_viol / safe_norms, 0.0)
        grad_bound = bound_factors[:, np.newaxis] * pts
        
        grad = grad_pairs + grad_bound
        
        return val, grad
    
    try:
        result = minimize(objective_and_grad, pts_init.reshape(-1), method='L-BFGS-B',
                          jac=True, options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-6})
        pts_opt = result.x.reshape(-1, 2)
        # Verify shape
        if pts_opt.shape != (N, 2):
            return pts_init, _score(pts_init)
        return pts_opt, _score(pts_opt)
    except Exception:
        return pts_init, _score(pts_init)
>>>>>>> REPLACE

I also need to fix the deadline.

