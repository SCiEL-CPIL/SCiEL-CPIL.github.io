import numpy as np
import time
import scipy.spatial
import scipy.optimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n=13 points on or inside a convex region to maximize
    the area of the smallest triangle formed by these points, normalized by the convex hull area.
    
    Returns:
        points: np.ndarray of shape (13, 2)
    """
    n = 13
    start_time = time.time()
    time_limit = 590  # Leave 40s margin from 630s

    # Precompute triangle indices
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_f, j_f, k_f = i_idx[mask], j_idx[mask], k_idx[mask]
    n_tris = len(i_f)

    def compute_ratio(points):
        points = np.asarray(points, dtype=float)
        if points.shape != (n, 2):
            return 0.0
        if np.any(~np.isfinite(points)):
            return 0.0
        try:
            hull = scipy.spatial.ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-12:
                return 0.0
        except Exception:
            return 0.0

        p1 = points[i_f]
        p2 = points[j_f]
        p3 = points[k_f]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        return min_area / A_hull

    def neg_objective(x):
        pts = x.reshape(n, 2)
        return -compute_ratio(pts)

    def smooth_objective(x, beta):
        """
        Smooth approximation of the negative ratio using log-sum-exp for soft-min.
        Returns (value, gradient) for L-BFGS-B.
        """
        pts = x.reshape(n, 2)
        
        # Compute convex hull area
        try:
            hull = scipy.spatial.ConvexHull(pts)
            A_hull = hull.volume
            if A_hull <= 1e-12:
                return 1e10, np.zeros_like(x)
        except Exception:
            return 1e10, np.zeros_like(x)
        
        # Compute triangle areas
        p1 = pts[i_f]
        p2 = pts[j_f]
        p3 = pts[k_f]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        # Soft-min using log-sum-exp: -logsumexp(-areas/beta)/beta
        # This approximates min(areas) as beta -> 0
        scaled_areas = areas / beta
        max_scaled = np.max(scaled_areas)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled_areas - max_scaled)))
        soft_min = -log_sum_exp * beta
        
        # The objective is to maximize soft_min / A_hull, so minimize -soft_min / A_hull
        value = -soft_min / A_hull
        
        # Gradient computation
        # d(soft_min)/d(area_i) = -exp(-area_i/beta) / sum(exp(-area_j/beta))
        neg_scaled = -areas / beta
        max_neg = np.max(neg_scaled)
        exp_neg = np.exp(neg_scaled - max_neg)
        sum_exp = np.sum(exp_neg)
        weights = exp_neg / sum_exp  # d(soft_min)/d(area_i)
        
        # d(area_i)/dx for each point
        # area_i = 0.5 * |cross_i|
        # For non-degenerate triangles, d(0.5*|cross|)/dx = 0.5*sign(cross)*d(cross)/dx
        sign_cross = np.sign(cross)
        grad_area = np.zeros_like(x)
        
        # For each triangle i, compute gradient of cross w.r.t. coordinates
        # cross = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
        # d(cross)/d(x1) = -(y3-y1), d(cross)/d(y1) = (x3-x1)
        # d(cross)/d(x2) = (y3-y1), d(cross)/d(y2) = -(x3-y1)
        # d(cross)/d(x3) = -(y2-y1), d(cross)/d(y3) = (x2-x1)
        
        for i in range(n_tris):
            w = weights[i] * sign_cross[i] * 0.5 / A_hull
            
            ii = i_f[i]
            jj = j_f[i]
            kk = k_f[i]
            
            x1, y1 = pts[ii]
            x2, y2 = pts[jj]
            x3, y3 = pts[kk]
            
            # Gradient of cross w.r.t. each point
            # Point 1
            grad_area[ii, 0] += -w * (y3 - y1)
            grad_area[ii, 1] += w * (x3 - x1)
            
            # Point 2
            grad_area[jj, 0] += w * (y3 - y1)
            grad_area[jj, 1] += -w * (x3 - x1)
            
            # Point 3
            grad_area[kk, 0] += -w * (y2 - y1)
            grad_area[kk, 1] += w * (x2 - x1)
        
        # The objective is -soft_min/A_hull, so gradient is -grad_area
        gradient = -grad_area
        
        return value, gradient

    # 1. Generate structured initial guesses
    guesses = []

    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    guesses.append(np.column_stack([np.cos(angles), np.sin(angles)]))

    # 12-gon + Center
    angles12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    poly12 = np.column_stack([np.cos(angles12), np.sin(angles12)])
    guesses.append(np.vstack([poly12, [[0.0, 0.0]]]))

    # 11-gon + Center + 1 point
    angles11 = np.linspace(0, 2 * np.pi, 11, endpoint=False)
    poly11 = np.column_stack([np.cos(angles11), np.sin(angles11)])
    guesses.append(np.vstack([poly11, [[0.0, 0.0]], [[1.0, 0.0]]]))

    # Two concentric hexagons (rotated) + Center
    # 6 + 6 + 1 = 13
    angles_h1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    poly_h1 = np.column_stack([np.cos(angles_h1), np.sin(angles_h1)])
    angles_h2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    poly_h2 = np.column_stack([0.5 * np.cos(angles_h2), 0.5 * np.sin(angles_h2)])
    guesses.append(np.vstack([poly_h1, poly_h2, [[0.0, 0.0]]]))

    # Random initializations (a few)
    rng = np.random.default_rng(seed=42)
    for _ in range(3):
        guesses.append(rng.normal(0, 1, (n, 2)))

    best_score = -np.inf
    best_points = None

    # Phase 1: Quick Nelder-Mead optimization on each guess
    for g in guesses:
        if time.time() - start_time > time_limit - 60:
            break
            
        try:
            x0 = g.flatten()
            res = scipy.optimize.minimize(
                neg_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_points = res.x.reshape(n, 2)
        except Exception:
            continue

    if best_points is None:
        # Fallback to regular 13-gon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = compute_ratio(best_points)

    # Phase 2: Refine with L-BFGS-B using smooth surrogate
    # Start with moderate beta and decrease
    betas = [0.1, 0.05, 0.02, 0.01]
    for beta in betas:
        if time.time() - start_time > time_limit - 30:
            break
        
        try:
            x0 = best_points.flatten()
            res = scipy.optimize.minimize(
                lambda x: smooth_objective(x, beta)[0],
                lambda x: smooth_objective(x, beta)[1],
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12}
            )
            # Evaluate with exact objective
            score = compute_ratio(res.x.reshape(n, 2))
            if score > best_score:
                best_score = score
                best_points = res.x.reshape(n, 2)
        except Exception:
            continue

    # Phase 3: Final high-precision polish with Nelder-Mead
    if time.time() - start_time < time_limit - 15:
        try:
            x0 = best_points.flatten()
            res = scipy.optimize.minimize(
                neg_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-13}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_points = res.x.reshape(n, 2)
        except Exception:
            pass

    # Phase 4: Random restarts with smooth refinement
    if time.time() - start_time < time_limit - 10:
        rng2 = np.random.default_rng(seed=99)
        for _ in range(3):
            if time.time() - start_time > time_limit - 5:
                break
            perturbed = best_points + rng2.normal(0, 0.03, (n, 2))
            x0 = perturbed.flatten()
            try:
                # Quick Nelder-Mead
                res_nm = scipy.optimize.minimize(
                    neg_objective,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                score_nm = -res_nm.fun
                
                # Then L-BFGS-B with small beta
                res_lbfgs = scipy.optimize.minimize(
                    lambda x: smooth_objective(x, 0.01)[0],
                    lambda x: smooth_objective(x, 0.01)[1],
                    res_nm.x,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                score_lbfgs = compute_ratio(res_lbfgs.x.reshape(n, 2))
                
                best_candidate = max(score_nm, score_lbfgs)
                if best_candidate > best_score:
                    best_score = best_candidate
                    if score_lbfgs >= score_nm:
                        best_points = res_lbfgs.x.reshape(n, 2)
                    else:
                        best_points = res_nm.x.reshape(n, 2)
            except Exception:
                continue

    return best_points
