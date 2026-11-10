import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 30.0
    N = 16
    N_PAIRS = N * (N - 1) // 2

    def get_ratio(pts):
        """Calculate d_min / d_max. Higher is better."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max <= 1e-12:
            return 0.0
        return d_min / d_max

    def get_ratio_sq(pts):
        """Calculate (d_min / d_max)^2. Higher is better. This is the actual metric R."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max <= 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def compute_distances(pts):
        """Compute all pairwise distances."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        dist = np.sqrt(np.maximum(dist_sq, 0.0))
        return dist

    def soft_min_max_objective(x, beta=10.0):
        """
        Smooth surrogate using log-sum-exp for min and max.
        Minimizes log(soft_max) - log(soft_min)
        """
        pts = x.reshape(N, 2)
        dist = compute_distances(pts)
        i_idx, j_idx = np.triu_indices(N, k=1)
        d_pairs = dist[i_idx, j_idx]
        
        if np.any(d_pairs < 1e-15):
            return 1e10, np.zeros_like(x)

        # Soft max: log(sum(exp(beta * d))) / beta
        # Soft min: -log(sum(exp(-beta * d))) / beta
        # To avoid overflow, shift by max
        d_max_val = np.max(d_pairs)
        d_min_val = np.min(d_pairs)
        
        # Soft max
        shifted_max = beta * (d_pairs - d_max_val)
        log_sum_exp_max = np.log(np.sum(np.exp(shifted_max)))
        soft_max_val = (np.log(np.sum(np.exp(shifted_max))) / beta + d_max_val)
        
        # Soft min
        shifted_min = -beta * (d_pairs - d_min_val)
        log_sum_exp_min = np.log(np.sum(np.exp(shifted_min)))
        soft_min_val = (-np.log(np.sum(np.exp(shifted_min))) / beta + d_min_val)
        
        if soft_min_val < 1e-15 or soft_max_val < 1e-15:
            return 1e10, np.zeros_like(x)
            
        obj_val = np.log(soft_max_val) - np.log(soft_min_val)
        
        # Gradient
        # d(soft_max)/dx = sum(exp(beta*d) * d(d)/dx) / sum(exp(beta*d))
        # d(soft_min)/dx = -sum(exp(-beta*d) * d(d)/dx) / sum(exp(-beta*d))
        
        w_max = np.exp(shifted_max)
        w_max /= np.sum(w_max)
        
        w_min = np.exp(shifted_min)
        w_min /= np.sum(w_min)
        
        grad = np.zeros_like(x)
        
        for k in range(N_PAIRS):
            i = i_idx[k]
            j = j_idx[k]
            d_ij = dist[i, j]
            if d_ij < 1e-15:
                continue
            diff_ij = pts[i] - pts[j]
            
            # Contribution to log(soft_max)
            grad[2*i] += w_max[k] * (diff_ij[0] / d_ij) / soft_max_val
            grad[2*i+1] += w_max[k] * (diff_ij[1] / d_ij) / soft_max_val
            grad[2*j] += w_max[k] * (-diff_ij[0] / d_ij) / soft_max_val
            grad[2*j+1] += w_max[k] * (-diff_ij[1] / d_ij) / soft_max_val
            
            # Contribution to -log(soft_min)
            grad[2*i] -= w_min[k] * (diff_ij[0] / d_ij) / soft_min_val
            grad[2*i+1] -= w_min[k] * (diff_ij[1] / d_ij) / soft_min_val
            grad[2*j] -= w_min[k] * (-diff_ij[0] / d_ij) / soft_min_val
            grad[2*j+1] -= w_min[k] * (-diff_ij[1] / d_ij) / soft_min_val
        
        return obj_val, grad

    def ring_init(n_inner, n_outer, r_inner, r_outer):
        """Points on two concentric rings."""
        pts = []
        if n_inner > 0:
            for i in range(n_inner):
                angle = 2 * np.pi * i / n_inner
                pts.append([r_inner * np.cos(angle), r_inner * np.sin(angle)])
        if n_outer > 0:
            for i in range(n_outer):
                angle = 2 * np.pi * i / n_outer + np.pi / n_outer
                pts.append([r_outer * np.cos(angle), r_outer * np.sin(angle)])
        return np.array(pts)

    def grid_init():
        """4x4 square grid."""
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i, j])
        pts = np.array(pts, dtype=float)
        pts = pts - np.mean(pts, axis=0)
        return pts

    def hex_init():
        """Hexagonal arrangement."""
        pts = []
        a = 1.0
        b = np.sqrt(3) / 2
        pts.append([0, 0])
        for i in range(6):
            angle = 2 * np.pi * i / 6
            pts.append([a * np.cos(angle), a * np.sin(angle)])
        ring2_pts = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * a + j * 0.5 * a
                y = j * b
                r = np.sqrt(x*x + y*y)
                if 1.5 * a - 0.1 < r < 2.5 * a + 0.1:
                    ring2_pts.append([x, y])
        ring2_pts = np.array(ring2_pts)
        if len(ring2_pts) >= 9:
            r_target = 2.0 * a
            dists_to_target = np.abs(np.linalg.norm(ring2_pts, axis=1) - r_target)
            idx = np.argsort(dists_to_target)[:9]
            ring2_pts = ring2_pts[idx]
        pts = np.array(pts)
        pts = np.vstack([pts, ring2_pts])
        pts = pts - np.mean(pts, axis=0)
        return pts

    def circle_init(n=16):
        """Points on a circle."""
        pts = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            pts.append([np.cos(angle), np.sin(angle)])
        return np.array(pts)

    def square_ring_init():
        """Points on boundary of a square with some interior."""
        pts = []
        for i in range(4):
            pts.append([-1 + 2*i/3, 1.0])
        for i in range(1, 4):
            pts.append([1.0, 1 - 2*i/3])
        for i in range(1, 4):
            pts.append([-1 + 2*i/3, -1.0])
        for i in range(1, 3):
            pts.append([-1.0, 1 - 2*i/3])
        pts.append([0, 0])
        pts.append([0.5, 0.5])
        pts.append([-0.5, 0.5])
        pts.append([0.5, -0.5])
        pts = np.array(pts[:16], dtype=float)
        pts = pts - np.mean(pts, axis=0)
        return pts

    def generate_candidates():
        candidates = []
        candidates.append(ring_init(1, 15, 0, 1.0))
        candidates.append(ring_init(4, 12, 0.5, 1.0))
        candidates.append(ring_init(7, 9, 0.4, 1.0))
        candidates.append(ring_init(9, 7, 0.4, 1.0))
        candidates.append(ring_init(1, 3, 0, 0.5))
        for i, c in enumerate(candidates):
            c = c - np.mean(c, axis=0)
            max_r = np.max(np.linalg.norm(c, axis=1))
            if max_r > 0:
                c = c / max_r
            candidates[i] = c
        candidates.append(grid_init())
        candidates.append(hex_init())
        candidates.append(circle_init(16))
        candidates.append(square_ring_init())
        return candidates

    best_pts = None
    best_ratio = 0.0

    for c in generate_candidates():
        if len(c) != 16:
            continue
        ratio = get_ratio(c)
        if ratio > best_ratio:
            best_ratio = ratio
            best_pts = c.copy()

    # Sequential Convex Programming with bisection on min distance
    def scp_optimize(pts, time_budget=300.0):
        """
        Sequential Convex Programming: minimize max distance subject to min distance >= t.
        Use bisection on t to find optimal packing.
        """
        pts = pts.copy()
        t_start = time.time()
        
        # Normalize to unit max distance
        pts = pts - np.mean(pts, axis=0)
        max_r = np.max(np.linalg.norm(pts, axis=1))
        if max_r > 0:
            pts = pts / max_r
        
        # Get initial ratio
        d = pdist(pts)
        d_min_init = np.min(d)
        d_max_init = np.max(d)
        ratio_init = d_min_init / d_max_init if d_max_init > 1e-12 else 0
        
        # Bisection on t (minimum distance constraint)
        # We want to find max t such that we can place points with all pairwise distances >= t
        # and minimize max distance
        
        t_low = 0.0
        t_high = ratio_init + 0.05  # Upper bound for feasible t
        
        # For each t, solve: minimize max_dist s.t. min_dist >= t
        # This is a convex problem (minimize max of convex functions s.t. convex constraints)
        # But max_dist is convex, min_dist >= t is non-convex constraint
        # Use SCP: iteratively linearize
        
        best_pts = pts.copy()
        best_ratio = ratio_init
        
        # Try several t values via bisection
        n_bisection = 12
        for b in range(n_bisection):
            if time.time() - t_start > time_budget:
                break
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            t_mid = (t_low + t_high) / 2
            
            # Solve SCP for given t_mid
            # Minimize max_dist s.t. all d_ij >= t_mid
            # Use penalty method: minimize max_dist + lambda * sum(min(0, d_ij - t_mid)^2)
            
            x0 = pts.flatten()
            lambda_penalty = 10.0
            
            def scp_objective(x, t_constraint, lam):
                pts_local = x.reshape(N, 2)
                dist = compute_distances(pts_local)
                i_idx, j_idx = np.triu_indices(N, k=1)
                d_pairs = dist[i_idx, j_idx]
                
                d_max_val = np.max(d_pairs)
                
                # Penalty for violating min distance constraint
                violations = np.maximum(0, t_constraint - d_pairs)
                penalty = lam * np.sum(violations ** 2)
                
                return d_max_val + penalty
            
            def scp_grad(x, t_constraint, lam):
                pts_local = x.reshape(N, 2)
                dist = compute_distances(pts_local)
                i_idx, j_idx = np.triu_indices(N, k=1)
                d_pairs = dist[i_idx, j_idx]
                
                d_max_val = np.max(d_pairs)
                
                # Gradient of max distance
                max_pairs = np.where(d_pairs > d_max_val - 1e-10)[0]
                
                # Penalty gradient
                violations = np.maximum(0, t_constraint - d_pairs)
                penalty_grad = np.zeros_like(x)
                
                for k in range(N_PAIRS):
                    i = i_idx[k]
                    j = j_idx[k]
                    d_ij = dist[i, j]
                    if d_ij < 1e-15:
                        continue
                    diff_ij = pts_local[i] - pts_local[j]
                    
                    # Gradient of max distance (if this pair is max)
                    if k in max_pairs:
                        penalty_grad[2*i] += (diff_ij[0] / d_ij) / len(max_pairs)
                        penalty_grad[2*i+1] += (diff_ij[1] / d_ij) / len(max_pairs)
                        penalty_grad[2*j] += (-diff_ij[0] / d_ij) / len(max_pairs)
                        penalty_grad[2*j+1] += (-diff_ij[1] / d_ij) / len(max_pairs)
                    
                    # Gradient of penalty
                    if violations[k] > 0:
                        penalty_grad[2*i] += 2 * lam * violations[k] * (-diff_ij[0] / d_ij)
                        penalty_grad[2*i+1] += 2 * lam * violations[k] * (-diff_ij[1] / d_ij)
                        penalty_grad[2*j] += 2 * lam * violations[k] * (diff_ij[0] / d_ij)
                        penalty_grad[2*j+1] += 2 * lam * violations[k] * (diff_ij[1] / d_ij)
                
                obj_val = d_max_val + lam * np.sum(violations ** 2)
                return obj_val, penalty_grad
            
            # SCP iterations
            x_current = x0.copy()
            for scp_iter in range(20):
                if time.time() - t_start > time_budget:
                    break
                if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                    break
                    
                try:
                    result = minimize(
                        lambda x: scp_objective(x, t_mid, lambda_penalty)[0],
                        x_current,
                        jac=lambda x: scp_grad(x, t_mid, lambda_penalty)[1],
                        method='L-BFGS-B',
                        options={'maxiter': 50, 'maxls': 30}
                    )
                    x_current = result.x
                except Exception:
                    break
            
            # Evaluate result
            pts_result = x_current.reshape(N, 2)
            d = pdist(pts_result)
            d_min = np.min(d)
            d_max = np.max(d)
            
            if d_max > 1e-12 and d_min >= t_mid - 1e-6:
                ratio = d_min / d_max
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pts = pts_result.copy()
                t_low = t_mid
            else:
                t_high = t_mid
        
        return best_pts, best_ratio

    # Run SCP optimization
    time_elapsed = time.time() - start_time
    scp_budget = min(300.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN)
    if scp_budget > 50:
        scp_pts, scp_ratio = scp_optimize(best_pts, time_budget=scp_budget)
        if scp_ratio > best_ratio:
            best_pts = scp_pts
            best_ratio = scp_ratio

    # Final polish with soft-min-max objective
    def soft_min_max_optimize(pts, time_budget=100.0):
        """Polish using smooth soft-min/max objective."""
        pts = pts.copy()
        t_start = time.time()
        
        x0 = pts.flatten()
        best_x = x0.copy()
        best_obj, _ = soft_min_max_objective(x0, beta=10.0)
        best_ratio = get_ratio(pts)
        
        n_restarts = 3
        for r in range(n_restarts):
            if time.time() - t_start > time_budget:
                break
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            if r == 0:
                x_start = x0.copy()
            else:
                x_start = x0 + np.random.randn(N * 2) * 0.05
            
            try:
                result = minimize(
                    lambda x: soft_min_max_objective(x, beta=10.0)[0],
                    x_start,
                    jac=lambda x: soft_min_max_objective(x, beta=10.0)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'maxls': 40, 'ftol': 1e-12}
                )
                obj_val, _ = soft_min_max_objective(result.x, beta=10.0)
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_x = result.x
                    best_ratio = get_ratio(result.x.reshape(N, 2))
            except Exception:
                pass
        
        return best_x.reshape(N, 2), best_ratio

    time_elapsed = time.time() - start_time
    polish_budget = min(100.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN)
    if polish_budget > 30:
        polish_pts, polish_ratio = soft_min_max_optimize(best_pts, time_budget=polish_budget)
        if polish_ratio > best_ratio:
            best_pts = polish_pts
            best_ratio = polish_ratio

    # Ensure finite and correct shape
    final_pts = best_pts
    if final_pts is None or not np.all(np.isfinite(final_pts)) or final_pts.shape != (16, 2):
        final_pts = grid_init()

    return final_pts
