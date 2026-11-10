import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    def raw_score(pts):
        """Compute (d_min / d_max)^2 for the point configuration."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, d) or not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_score(pts, beta=20.0):
        """
        Smooth surrogate for (d_min / d_max)^2.
        Uses soft-min and soft-max approximations.
        """
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, d) or not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        
        # Soft min: -1/beta * log(sum(exp(-beta * d)))
        # Soft max: 1/beta * log(sum(exp(beta * d)))
        # To avoid overflow/underflow, we can shift by max/min
        d_min_soft = -np.log(np.sum(np.exp(-beta * dists))) / beta
        d_max_soft = np.log(np.sum(np.exp(beta * dists))) / beta
        
        if d_max_soft < 1e-12:
            return 0.0
            
        ratio = d_min_soft / d_max_soft
        return ratio ** 2

    def neg_smooth_score(pts, beta=20.0):
        return -smooth_score(pts, beta)

    # 1. Global Search with Differential Evolution
    # We optimize the smooth score.
    # Bounds: Points should be within a reasonable range. 
    # Since the metric is scale invariant, we can fix the scale by normalizing 
    # inside the objective function or just let DE explore a box.
    # Let's normalize the points inside the objective to keep them in a compact space.
    
    def normalized_neg_smooth_score(flat_pts, beta=20.0):
        pts = flat_pts.reshape(n, d)
        # Normalize: center at 0, scale such that max norm is 1
        center = np.mean(pts, axis=0)
        pts = pts - center
        norms = np.linalg.norm(pts, axis=1)
        max_norm = np.max(norms)
        if max_norm < 1e-12:
            return 0.0
        pts = pts / max_norm
        return neg_smooth_score(pts, beta)

    # Initial bounds for DE
    bounds = [(-5.0, 5.0)] * (n * d)
    
    best_score = -np.inf
    best_points = None

    # Run DE
    try:
        # maxiter and popsize control the search intensity
        # We have ~500s budget. DE is slow but robust.
        # Let's run DE for a moderate number of iterations.
        res = differential_evolution(
            normalized_neg_smooth_score,
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            seed=42,
            init='latinhypercube',
            polish=False # We will do our own polishing
        )
        
        if res.success or res.fun < 0: # res.fun is negative score
            pts_candidate = res.x.reshape(n, d)
            # Re-normalize
            center = np.mean(pts_candidate, axis=0)
            pts_candidate = pts_candidate - center
            norms = np.linalg.norm(pts_candidate, axis=1)
            max_norm = np.max(norms)
            if max_norm > 0:
                pts_candidate = pts_candidate / max_norm
            
            s = raw_score(pts_candidate)
            if s > best_score:
                best_score = s
                best_points = pts_candidate.copy()
    except Exception:
        pass

    # If DE failed or timed out, use a fallback initialization
    if best_points is None:
        # Fallback: Fibonacci sphere
        rng = np.random.RandomState(42)
        indices = np.arange(0, n, dtype=float) + 0.5
        phi_f = np.arccos(1 - 2 * indices / n)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.cos(theta) * np.sin(phi_f)
        y = np.sin(theta) * np.sin(phi_f)
        z = np.cos(phi_f)
        best_points = np.column_stack([x, y, z])
        best_score = raw_score(best_points)

    # 2. Local Refinement with Nelder-Mead on the smooth score
    # We refine the best point found by DE.
    
    def refine_obj(flat_pts, beta=50.0):
        pts = flat_pts.reshape(n, d)
        # Normalize inside to keep search space compact
        center = np.mean(pts, axis=0)
        pts = pts - center
        norms = np.linalg.norm(pts, axis=1)
        max_norm = np.max(norms)
        if max_norm < 1e-12:
            return 0.0
        pts = pts / max_norm
        return neg_smooth_score(pts, beta)

    # Run NM from the best point and a few perturbations
    rng = np.random.RandomState(123)
    starts = [best_points.copy().flatten()]
    for i in range(5):
        perturbed = best_points + rng.randn(n, d) * 0.1
        # Normalize perturbed
        center = np.mean(perturbed, axis=0)
        perturbed = perturbed - center
        norms = np.linalg.norm(perturbed, axis=1)
        max_norm = np.max(norms)
        if max_norm > 0:
            perturbed = perturbed / max_norm
        starts.append(perturbed.flatten())

    for start in starts:
        if time.time() > deadline - 60:
            break
        try:
            res_nm = minimize(
                refine_obj,
                start,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            pts_candidate = res_nm.x.reshape(n, d)
            # Normalize
            center = np.mean(pts_candidate, axis=0)
            pts_candidate = pts_candidate - center
            norms = np.linalg.norm(pts_candidate, axis=1)
            max_norm = np.max(norms)
            if max_norm > 0:
                pts_candidate = pts_candidate / max_norm
            
            s = raw_score(pts_candidate)
            if s > best_score:
                best_score = s
                best_points = pts_candidate.copy()
        except Exception:
            pass

    # 3. Final Polishing: Directly optimize the raw metric with small steps
    # This helps if the smooth approximation missed the exact peak.
    if time.time() < deadline - 10:
        pts = best_points.copy()
        for i in range(100):
            if time.time() > deadline - 5:
                break
            dists = pdist(pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max < 1e-12:
                break
            current_score = (d_min / d_max) ** 2
            
            # Find min and max pairs
            min_idx = np.argmin(dists)
            max_idx = np.argmax(dists)
            
            # Convert indices to pairs
            count = 0
            i_min = j_min = -1
            for i in range(n):
                for j in range(i + 1, n):
                    if count == min_idx:
                        i_min = i
                        j_min = j
                        break
                    count += 1
                if i_min != -1:
                    break
            
            count = 0
            i_max = j_max = -1
            for i in range(n):
                for j in range(i + 1, n):
                    if count == max_idx:
                        i_max = i
                        j_max = j
                        break
                    count += 1
                if i_max != -1:
                    break

            # Move min pair apart
            diff_min = pts[i_min] - pts[j_min]
            norm_min = np.linalg.norm(diff_min)
            if norm_min > 1e-12:
                dir_min = diff_min / norm_min
                step = 0.01 * norm_min
                pts[i_min] += dir_min * step
                pts[j_min] -= dir_min * step
            
            # Move max pair closer
            diff_max = pts[i_max] - pts[j_max]
            norm_max = np.linalg.norm(diff_max)
            if norm_max > 1e-12:
                dir_max = diff_max / norm_max
                step = 0.01 * norm_max
                pts[i_max] -= dir_max * step
                pts[j_max] += dir_max * step

            s_new = raw_score(pts)
            if s_new > current_score:
                if s_new > best_score:
                    best_score = s_new
                    best_points = pts.copy()
            else:
                pts = best_points.copy()

    # Final normalization for output
    best_points = best_points - np.mean(best_points, axis=0)
    norm_max = np.max(np.linalg.norm(best_points, axis=1))
    if norm_max > 0:
        best_points = best_points / norm_max * 100

    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))

    return best_points
