"""Optimized planar point configuration for maximum product of distances at fixed diameter."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(points):
    """Compute Delta/n^n for 8 points, scaled to diameter 2."""
    pts = np.array(points, dtype=float)
    n = len(pts)
    # Compute all pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    # Diameter
    diam = np.max(dists)
    if diam < 1e-12:
        return 0.0
    # Scale to diameter 2
    scale = 2.0 / diam
    scaled_dists = dists * scale
    # Product of all ordered pair distances (i != j)
    # Product of distances over ordered pairs = product over i<j of d_ij^2
    # Since ordered pairs (i,j) for i!=j, that's 2 * (number of unordered pairs)
    # Product over ordered pairs = (product over unordered pairs) ^ 2
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    prod_ordered = 1.0
    for i in range(n):
        for j in range(i+1, n):
            prod_ordered *= scaled_dists[i, j] ** 2
    delta = prod_ordered
    score = delta / (n ** n)
    return score

def _neg_log_score(x):
    """Negative log of score for minimization. x is 16-dim flat array."""
    pts = x.reshape(-1, 2)
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    diam = np.max(dists)
    if diam < 1e-12:
        return 1e10
    scale = 2.0 / diam
    # Log of product of ordered pair distances
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    log_prod = 0.0
    for i in range(n):
        for j in range(i+1, n):
            log_prod += 2.0 * np.log(scaled_dists[i, j]) if scaled_dists[i,j] > 1e-15 else -1e10
    # score = exp(log_prod) / n^n
    # log(score) = log_prod - n*log(n)
    # neg_log_score = -log(score) = n*log(n) - log_prod
    if scaled_dists.min() < 1e-15:
        return 1e10
    log_prod = 0.0
    for i in range(n):
        for j in range(i+1, n):
            log_prod += 2.0 * np.log(scaled_dists[i, j])
    return n * np.log(n) - log_prod

def construct():
    deadline = time.time() + 148  # Leave 2s safety margin
    n = 8
    
    # Start with regular octagon
    k_vals = np.arange(n)
    base_angle = 2 * np.pi / n
    octagon = np.column_stack([np.cos(base_angle * k_vals), np.sin(base_angle * k_vals)])
    
    best_points = octagon.copy()
    best_score = _compute_score(octagon)
    
    iu = np.triu_indices(n, k=1)
    c = n * (n - 1) // 2
    
    # Smooth surrogate: use soft-max (log-sum-exp) for diameter to get a smooth objective
    # This avoids the non-smooth max() in the exact objective
    def neg_log_score_smooth(x, beta=200.0):
        """Negative log score with soft-max diameter approximation.
        beta controls sharpness of soft-max; large beta approximates true max."""
        pts = x.reshape(-1, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        # Soft-max of squared distances (upper triangle only, excluding diagonal)
        dsq_upper = dists_sq[iu]
        if np.min(dsq_upper) < 1e-30:
            return 1e10
        # log-sum-exp for soft-max: log(sum(exp(dsq/beta))) * beta
        # Use numerical stability: subtract max
        m = np.max(dsq_upper)
        log_sum_exp = m + beta * np.log(np.sum(np.exp((dsq_upper - m) / beta)))
        diam_sq_soft = log_sum_exp
        if diam_sq_soft < 1e-24:
            return 1e10
        log_dsq = np.log(dsq_upper)
        log_sum_dsq = np.sum(log_dsq)
        log_diam_sq = np.log(diam_sq_soft)
        # log_prod_ordered = 2*c*log(2) + sum_{i<j} log(dsq_ij) - c*log(diam_sq)
        log_prod_ordered = 2.0 * c * np.log(2.0) + log_sum_dsq - c * log_diam_sq
        return n * np.log(n) - log_prod_ordered
    
    # Exact (non-smooth) objective for final polish
    def neg_log_score_exact(x):
        pts = x.reshape(-1, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        dsq_upper = dists_sq[iu]
        diam_sq = np.max(dsq_upper)
        if diam_sq < 1e-24 or np.min(dsq_upper) < 1e-30:
            return 1e10
        log_sum_dsq = np.sum(np.log(dsq_upper))
        log_diam_sq = np.log(diam_sq)
        log_prod_ordered = 2.0 * c * np.log(2.0) + log_sum_dsq - c * log_diam_sq
        return n * np.log(n) - log_prod_ordered
    
    # Gradient of the exact objective (for gradient-based polish on the kinked function)
    def neg_log_score_exact_grad(x):
        pts = x.reshape(-1, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        dsq_upper = dists_sq[iu]
        diam_sq = np.max(dsq_upper)
        if diam_sq < 1e-24 or np.min(dsq_upper) < 1e-30:
            return 1e10, np.zeros(2 * n)
        log_sum_dsq = np.sum(np.log(dsq_upper))
        log_diam_sq = np.log(diam_sq)
        log_prod_ordered = 2.0 * c * np.log(2.0) + log_sum_dsq - c * log_diam_sq
        fval = n * np.log(n) - log_prod_ordered
        
        # Gradient: d/dx_k of neg_log_score = d/dx_k [n*log(n) - 2*c*log(2) - sum_{i<j} log(dsq_ij) + c*log(diam_sq)]
        # = -sum_{i<j} d(log(dsq_ij))/dx_k + c * d(log(diam_sq))/dx_k
        # d(log(dsq_ij))/dx_k = 2*(pts_i - pts_j)_k / dsq_ij  (for the k-th coordinate of point i or j)
        # For diam_sq: only the max pair contributes (subgradient)
        
        grad = np.zeros(2 * n)
        # Contribution from -sum log(dsq_ij)
        for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
            dsq = dsq_upper[idx]
            if dsq < 1e-30:
                continue
            diff = pts[i] - pts[j]
            # d log(dsq_ij) / d pts[i] = 2*diff/dsq
            # d log(dsq_ij) / d pts[j] = -2*diff/dsq
            # neg_log has -log(dsq), so gradient contribution is -2*diff/dsq for pts[i], +2*diff/dsq for pts[j]
            grad[2*i] -= 2.0 * diff[0] / dsq
            grad[2*i+1] -= 2.0 * diff[1] / dsq
            grad[2*j] += 2.0 * diff[0] / dsq
            grad[2*j+1] += 2.0 * diff[1] / dsq
        
        # Contribution from c*log(diam_sq): subgradient from the max pair(s)
        # Find all pairs achieving the max
        max_idx = np.where(np.abs(dsq_upper - diam_sq) < 1e-12)[0]
        if len(max_idx) > 0:
            # Average subgradient over all max pairs
            for idx in max_idx:
                i, j = iu[0][idx], iu[1][idx]
                diff = pts[i] - pts[j]
                # d log(diam_sq) / d pts[i] = 2*diff/diam_sq
                # d log(diam_sq) / d pts[j] = -2*diff/diam_sq
                # neg_log has +c*log(diam_sq), so gradient is +c*2*diff/diam_sq for pts[i], -c*2*diff/diam_sq for pts[j]
                grad[2*i] += c * 2.0 * diff[0] / diam_sq
                grad[2*i+1] += c * 2.0 * diff[1] / diam_sq
                grad[2*j] -= c * 2.0 * diff[0] / diam_sq
                grad[2*j+1] -= c * 2.0 * diff[1] / diam_sq
            # Normalize by number of max pairs for a proper subgradient
            grad /= len(max_idx)
        
        return fval, grad
    
    def check_and_update(pts):
        nonlocal best_points, best_score
        min_dist = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                min_dist = min(min_dist, d)
        if min_dist > 1e-6:
            sc = _compute_score(pts)
            if sc > best_score:
                best_score = sc
                best_points = pts.copy()
        return min_dist
    
    # Multiple random restarts with increasing perturbation
    rng = np.random.default_rng(42)
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() > deadline - 40:
            break
        
        # Varying noise scales: start small, then larger
        if r < 5:
            noise_scale = 0.05 * (r + 1)
        elif r < 10:
            noise_scale = 0.3 + 0.1 * (r - 5)
        else:
            noise_scale = 0.8 + 0.15 * (r - 10)
        
        x0 = octagon.flatten() + rng.normal(0, noise_scale, size=2 * n)
        
        try:
            from scipy.optimize import minimize
            # Use smooth surrogate for L-BFGS (needs gradients)
            res = minimize(neg_log_score_smooth, x0, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < 1e6:
                pts = res.x.reshape(-1, 2)
                check_and_update(pts)
        except Exception:
            pass
    
    # Nelder-Mead polish on exact score from best point found
    if time.time() < deadline - 25:
        try:
            from scipy.optimize import minimize
            res = minimize(neg_log_score_exact, best_points.flatten(), method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
            pts = res.x.reshape(-1, 2)
            check_and_update(pts)
        except Exception:
            pass
    
    # Second round of restarts from the improved best point
    if time.time() < deadline - 30:
        for r in range(8):
            if time.time() > deadline - 20:
                break
            noise_scale = 0.05 + 0.05 * r
            x0 = best_points.flatten() + rng.normal(0, noise_scale, size=2 * n)
            try:
                from scipy.optimize import minimize
                res = minimize(neg_log_score_smooth, x0, method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.fun < 1e6:
                    pts = res.x.reshape(-1, 2)
                    check_and_update(pts)
            except Exception:
                pass
    
    # Gradient-based polish on exact objective using analytic gradient
    if time.time() < deadline - 15:
        try:
            from scipy.optimize import minimize
            # Use L-BFGS-B with analytic gradient on the exact (kinked) objective
            res = minimize(neg_log_score_exact_grad, best_points.flatten(),
                          method='L-BFGS-B', jac=True,
                          options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < 1e6:
                pts = res.x.reshape(-1, 2)
                check_and_update(pts)
        except Exception:
            pass
    
    # Final high-precision Nelder-Mead polish
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize
            res = minimize(neg_log_score_exact, best_points.flatten(), method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16})
            pts = res.x.reshape(-1, 2)
            check_and_update(pts)
        except Exception:
            pass
    
    # Coordinate descent refinement: optimize each coordinate independently
    if time.time() < deadline - 5:
        for iteration in range(50):
            if time.time() > deadline - 3:
                break
            improved = False
            for coord_idx in range(2 * n):
                # Save current value
                old_val = best_points.flatten()[coord_idx]
                # Try small perturbations
                for delta in [1e-4, 1e-5, 1e-6, -1e-4, -1e-5, -1e-6, 1e-3, -1e-3]:
                    new_val = old_val + delta
                    if abs(new_val) >= 1e6:
                        continue
                    test_pts = best_points.copy()
                    flat = test_pts.flatten()
                    flat[coord_idx] = new_val
                    test_pts = flat.reshape(-1, 2)
                    sc = _compute_score(test_pts)
                    if sc > best_score + 1e-15:
                        best_score = sc
                        best_points = test_pts
                        improved = True
                        break
            if not improved:
                break
    
    # Final verification
    sc = _compute_score(best_points)
    if sc <= 0:
        return [[float(octagon[i, 0]), float(octagon[i, 1])] for i in range(n)]
    
    # Check coordinates
    if np.max(np.abs(best_points)) >= 1e6:
        return [[float(octagon[i, 0]), float(octagon[i, 1])] for i in range(n)]
    
    return [[float(best_points[i, 0]), float(best_points[i, 1])] for i in range(n)]
# EVOLVE-BLOCK-END
