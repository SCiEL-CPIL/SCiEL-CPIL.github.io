import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 610  # 10s margin before 620s hard limit (evaluator kills at 630s)
    
    best_score = -1.0
    best_points = np.zeros((n, d))
    
    # Helper to calculate the raw metric score
    def get_score(pts):
        dists = pdist(pts, metric='sqeuclidean')
        if len(dists) == 0:
            return 0.0
        d_min_sq = np.min(dists)
        d_max_sq = np.max(dists)
        
        if d_max_sq < 1e-12:
            return 0.0
        
        return d_min_sq / d_max_sq

    # Smooth objective function using soft-min and soft-max approximations
    # We want to maximize d_min / d_max.
    # Let f(x) = soft_min(dist_sq) - beta * (soft_max(dist_sq) - 1)^2
    # Actually, simpler: Maximize soft_min(dist_sq) subject to soft_max(dist_sq) ~ 1.
    # Or: Maximize soft_min(dist_sq) / soft_max(dist_sq).
    # The ratio is scale invariant. Let's fix scale by penalizing deviation of max dist from 1.
    
    def objective(x, beta=50.0):
        pts = x.reshape(n, d)
        dists_sq = pdist(pts, metric='sqeuclidean')
        
        # Soft min: approximates min(dists_sq)
        # -1/beta * log(sum(exp(-beta * dists_sq)))
        # For large beta, this is close to min.
        # Need to be careful with numerical stability.
        min_dists = np.min(dists_sq)
        exp_terms = np.exp(-beta * (dists_sq - min_dists))
        soft_min = min_dists - (1.0 / beta) * np.log(np.sum(exp_terms))
        
        # Soft max: approximates max(dists_sq)
        # 1/beta * log(sum(exp(beta * dists_sq)))
        max_dists = np.max(dists_sq)
        exp_terms_max = np.exp(beta * (dists_sq - max_dists))
        soft_max = max_dists + (1.0 / beta) * np.log(np.sum(exp_terms_max))
        
        # We want to maximize soft_min while keeping soft_max close to 1.
        # If soft_max > 1, we penalize. If < 1, we don't care as much (can scale up).
        # Actually, since the metric is scale invariant, we can just maximize soft_min / soft_max.
        # But division can be unstable.
        # Let's try: maximize soft_min - lambda * max(0, soft_max - 1)^2 - lambda * max(0, 1 - soft_max)^2?
        # No, simpler: The ratio r = d_min / d_max.
        # If we normalize such that d_max = 1, we maximize d_min.
        # Let's use a penalty for d_max != 1.
        penalty = (soft_max - 1.0) ** 2
        
        # We want to maximize soft_min. So minimize -soft_min + penalty_weight * penalty
        return -soft_min + penalty

    def gradient(x, beta=50.0):
        pts = x.reshape(n, d)
        # Compute pairwise squared distances matrix
        # dist_sq[i, j] = ||p_i - p_j||^2
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq_mat = np.sum(diff * diff, axis=2)
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        
        # Get vector of squared distances for soft min/max
        dists_sq = dists_sq_mat[mask]
        
        # Soft min gradient w.r.t dists_sq
        min_dists = np.min(dists_sq)
        exp_terms = np.exp(-beta * (dists_sq - min_dists))
        sum_exp = np.sum(exp_terms)
        # d(soft_min)/d(dist_sq_k) = (1/sum_exp) * exp(-beta*(dist_sq_k - min_dists)) * (-beta) * (1 - delta(k, argmin? no, just derivative of log sum exp))
        # Let S = sum(exp(-beta * (d_i - m))). soft_min = m - 1/beta * log(S).
        # d(soft_min)/d(d_k) = -delta_{k, argmin} (if m changes) - 1/beta * (1/S) * dS/dd_k
        # This is tricky because min_dists is non-differentiable.
        # However, the standard log-sum-exp gradient is smooth.
        # Let's use the standard formula for the gradient of the log-sum-exp function.
        # g_i = exp(-beta * (d_i - m)) / S
        # Then d(soft_min)/d(d_i) = -g_i
        g_min = exp_terms / sum_exp
        
        # Soft max gradient w.r.t dists_sq
        max_dists = np.max(dists_sq)
        exp_terms_max = np.exp(beta * (dists_sq - max_dists))
        sum_exp_max = np.sum(exp_terms_max)
        # soft_max = m + 1/beta * log(S_max)
        # d(soft_max)/d(d_i) = g_max_i
        g_max = exp_terms_max / sum_exp_max
        
        # Penalty term: (soft_max - 1)^2
        # d(penalty)/d(d_i) = 2 * (soft_max - 1) * g_max_i
        
        # Total objective: -soft_min + (soft_max - 1)^2
        # d(obj)/d(d_i) = -(-g_min_i) + 2 * (soft_max - 1) * g_max_i
        # = g_min_i + 2 * (soft_max - 1) * g_max_i
        
        grad_dists = g_min + 2.0 * (max_dists + (1.0/beta)*np.log(sum_exp_max) - 1.0) * g_max
        
        # Map gradient from pairwise distances to points
        # dist_sq[i, j] = ||p_i - p_j||^2
        # d(dist_sq[i, j])/d(p_i) = 2 * (p_i - p_j)
        # d(dist_sq[i, j])/d(p_j) = 2 * (p_j - p_i)
        
        # Create gradient matrix
        grad_mat = np.zeros((n, n, d))
        
        # We have grad_dists for the flattened upper triangle (or all pairs?).
        # pdist returns condensed distance vector.
        # It's easier to compute the gradient for the full matrix and then take the upper triangle or sum appropriately.
        # Let's compute grad for full symmetric matrix.
        
        # We need to map grad_dists (length n*(n-1)/2) back to matrix.
        # Or, simpler: compute the gradient for the full matrix directly in the objective?
        # No, let's stick to the condensed vector.
        
        # Indices for condensed vector
        i_idx, j_idx = np.triu_indices(n, k=1)
        
        # Initialize full gradient matrix for distances
        grad_dists_mat = np.zeros((n, n))
        grad_dists_mat[i_idx, j_idx] = grad_dists
        grad_dists_mat[j_idx, i_idx] = grad_dists
        
        # Now sum contributions to each point
        # For point k:
        # grad_p_k = sum_{j != k} (d(dist_sq[k,j])/d(p_k) * grad_dists_mat[k,j] + d(dist_sq[j,k])/d(p_k) * grad_dists_mat[j,k])
        # Since dist_sq[k,j] = dist_sq[j,k] and matrix is symmetric, and grad_dists_mat is symmetric:
        # d(dist_sq[k,j])/d(p_k) = 2(p_k - p_j)
        # So term is 2 * grad_dists_mat[k,j] * (p_k - p_j)
        # Sum over j.
        
        # Vectorized:
        # For each k, sum over j: 2 * grad_dists_mat[k, j] * (p_k - p_j)
        # Let G be grad_dists_mat.
        # grad_p_k = 2 * sum_j G[k,j] p_k - 2 * sum_j G[k,j] p_j
        # = 2 * p_k * sum_j G[k,j] - 2 * (G @ p)_k
        
        row_sums = np.sum(grad_dists_mat, axis=1)
        Gp = grad_dists_mat @ pts
        
        grad_pts = 2.0 * (pts * row_sums[:, np.newaxis] - Gp)
        
        return grad_pts.flatten()

    def run_l_bfgs(start_pts, beta=50.0):
        x0 = start_pts.flatten()
        
        def obj(x):
            return objective(x, beta=beta)
            
        def grad(x):
            return gradient(x, beta=beta)

        try:
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            return res.x.reshape(n, d)
        except Exception:
            return start_pts

    # Initial points: Random on sphere
    def get_random_points():
        pts = np.random.randn(n, d)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 10:
        start_pts = get_random_points()
        
        # Try a few different beta values to balance smoothness and accuracy
        betas = [20.0, 50.0, 100.0]
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
            
            local_best = run_l_bfgs(start_pts, beta=beta)
            local_score = get_score(local_best)
            
            if local_score > best_score:
                best_score = local_score
                best_points = local_best.copy()
                
            # If we have a very good score, we can try to refine with higher beta
            if best_score > 0.9:
                 # Refine
                 refined = run_l_bfgs(best_points, beta=200.0)
                 ref_score = get_score(refined)
                 if ref_score > best_score:
                     best_score = ref_score
                     best_points = refined.copy()
        
        num_restarts += 1
        
        # If we've found a near-perfect solution, break
        if best_score > 0.99:
            break

    # Final check
    if best_score < 0:
        # Fallback to random if optimization failed completely
        best_points = get_random_points()
        best_score = get_score(best_points)

    return best_points
