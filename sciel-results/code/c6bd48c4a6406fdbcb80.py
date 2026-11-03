import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13
TIME_LIMIT = 145.0  # Leave 5s margin before 150s kill

def construct():
    start_time = time.time()
    time_limit = TIME_LIMIT

    # Precompute triple index arrays for vectorized area computation
    _tri_i = []
    _tri_j = []
    _tri_k = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                _tri_i.append(i)
                _tri_j.append(j)
                _tri_k.append(k)
    tri_i = np.array(_tri_i, dtype=np.int32)
    tri_j = np.array(_tri_j, dtype=np.int32)
    tri_k = np.array(_tri_k, dtype=np.int32)
    n_triples = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area for given points."""
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0, 0.0

        if area_hull < 1e-9:
            return 0.0, 0.0

        x = points[:, 0]
        y = points[:, 1]
        # Vectorized cross products for all 286 triples
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        min_area = float(np.min(areas))

        score = min_area / area_hull
        return score, area_hull

    def _soft_min_score(points, beta=50.0):
        """
        Smooth surrogate: use log-sum-exp to approximate the minimum triangle area.
        Returns negative soft-min score for minimization.
        """
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 1e6

        if area_hull < 1e-9:
            return 1e6

        x = points[:, 0]
        y = points[:, 1]

        # Vectorized triangle areas
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)

        # Soft-min via log-sum-exp (stable)
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta

        score = soft_min_area / area_hull
        return -score

    def _powell_objective(points_flat):
        """Objective for Powell method: negative of hard score."""
        points = points_flat.reshape(-1, 2)
        score, _ = compute_score(points)
        return -score

    def _hard_score_neg(points_flat):
        """Negative hard score for final evaluation."""
        points = points_flat.reshape(-1, 2)
        score, _ = compute_score(points)
        return -score

    def _soft_objective(points_flat, beta=50.0):
        """Soft-min objective for gradient-based optimization."""
        points = points_flat.reshape(-1, 2)
        return _soft_min_score(points, beta)

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        """Generate points of a regular n-gon."""
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _ring_config(r1, r2, r3, n1, n2, n3, phase1=0.0, phase2=0.0, phase3=0.0):
        """
        Symmetric ring configuration with 3 rings.
        n1 points on ring 1 (radius r1), n2 on ring 2 (radius r2), n3 on ring 3 (radius r3).
        If n3==1, it's a single center point at (0,0).
        """
        pts = []
        if n1 > 0:
            a = np.linspace(phase1, phase1 + 2 * np.pi, n1, endpoint=False)
            pts.append(np.column_stack([r1 * np.cos(a), r1 * np.sin(a)]))
        if n2 > 0:
            a = np.linspace(phase2, phase2 + 2 * np.pi, n2, endpoint=False)
            pts.append(np.column_stack([r2 * np.cos(a), r2 * np.sin(a)]))
        if n3 == 1:
            pts.append(np.array([[0.0, 0.0]]))
        elif n3 > 1:
            a = np.linspace(phase3, phase3 + 2 * np.pi, n3, endpoint=False)
            pts.append(np.column_stack([r3 * np.cos(a), r3 * np.sin(a)]))
        if not pts:
            return _regular_polygon(N, radius=1.0)
        return np.vstack(pts)

    def _local_search(points, time_limit):
        """
        Direct local search: iteratively move points to improve the minimum triangle area.
        Focuses on the worst triangles and makes fine-grained improvements.
        """
        pts = points.copy()
        best_s, _ = compute_score(pts)
        
        # Precompute which triples involve each point
        point_triples = [[] for _ in range(N)]
        for idx in range(n_triples):
            for p in [tri_i[idx], tri_j[idx], tri_k[idx]]:
                point_triples[p].append(idx)
        
        # Find the worst triangle
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        
        # Try moving each point to improve the worst triangles
        for iter_num in range(500):
            if time.time() > time_limit:
                break
            
            # Find the worst triangle
            min_idx = np.argmin(areas)
            min_area = areas[min_idx]
            
            # Try moving each point in the worst triangle
            for p in [tri_i[min_idx], tri_j[min_idx], tri_k[min_idx]]:
                # Try small perturbations
                for dx in [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]:
                    for dy in [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]:
                        if dx == 0.0 and dy == 0.0:
                            continue
                        trial = pts.copy()
                        trial[p, 0] += dx
                        trial[p, 1] += dy
                        
                        # Quick check: compute areas for triples involving p
                        trial_x = trial[:, 0]
                        trial_y = trial[:, 1]
                        trial_v1x = trial_x[tri_j] - trial_x[tri_i]
                        trial_v1y = trial_y[tri_j] - trial_y[tri_i]
                        trial_v2x = trial_x[tri_k] - trial_x[tri_i]
                        trial_v2y = trial_y[tri_k] - trial_y[tri_i]
                        trial_areas = 0.5 * np.abs(trial_v1x * trial_v2y - trial_v1y * trial_v2x)
                        trial_min = np.min(trial_areas)
                        
                        if trial_min > min_area:
                            # Accept and update
                            pts = trial
                            areas = trial_areas
                            min_area = trial_min
                            min_idx = np.argmin(areas)
                            break
                    if time.time() > time_limit:
                        break
                    if min_area > areas[min_idx]:
                        break
            
            # Check if we improved the overall score
            s, _ = compute_score(pts)
            if s > best_s:
                best_s = s
        
        return pts, best_s

    def _optimize_from(start_pts, time_budget):
        """Run multi-stage optimization from a starting configuration."""
        best = start_pts.copy()
        best_s, _ = compute_score(best)
        current = start_pts.copy()

        # Stage 1: L-BFGS-B with annealed soft-min beta
        for beta in [10.0, 25.0, 60.0, 120.0, 250.0]:
            if time.time() - start_time > time_budget:
                break
            try:
                res = opt.minimize(
                    lambda x: _soft_min_score(x.reshape(-1, 2), beta),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
                current = cand
            except Exception:
                pass

        # Stage 2: Direct local search targeting worst triangles
        if time.time() - start_time < time_budget - 8:
            try:
                cand, s = _local_search(best, time.time() - start_time + (time_budget - time.time() - start_time) * 0.5)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

        # Stage 3: Powell method for robust derivative-free refinement
        if time.time() - start_time < time_budget - 5:
            try:
                res = opt.minimize(
                    _powell_objective,
                    best.flatten(),
                    method='Powell',
                    options={'maxiter': 150, 'ftol': 1e-12, 'xtol': 1e-10}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

        # Stage 4: Nelder-Mead polish on hard objective
        if time.time() - start_time < time_budget - 3:
            try:
                res = opt.minimize(
                    lambda x: -compute_score(x.reshape(-1, 2))[0],
                    best.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'adaptive': True, 'xatol': 1e-11, 'fatol': 1e-13}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

        return best, best_s

    def _make_config(k_hull, k_inner):
        """
        Create a configuration with k_hull points on a regular polygon
        and k_inner points arranged inside.
        """
        hull_pts = _regular_polygon(k_hull, radius=1.0)
        if k_inner == 0:
            return hull_pts

        if k_inner == 1:
            inner_pts = np.array([[0.0, 0.0]])
        else:
            inner_pts = _regular_polygon(k_inner, radius=0.3, center=(0.0, 0.0))

        return np.vstack([hull_pts, inner_pts])

    best_points = None
    best_score = -float('inf')

    # --- Phase 1: Symmetric ring initializations (low-dimensional, smooth landscape) ---
    # These capture the geometry that the cell findings indicate is near-optimal.
    ring_starts = [
        # 1 center + 12 outer
        _ring_config(r1=1.0, r2=0.0, r3=0.0, n1=12, n2=0, n3=1),
        # 1 center + 6 mid + 6 outer
        _ring_config(r1=1.0, r2=0.5, r3=0.0, n1=6, n2=6, n3=1),
        # 1 center + 5 mid + 7 outer (asymmetric split)
        _ring_config(r1=1.0, r2=0.45, r3=0.0, n1=7, n2=5, n3=1),
        # 1 center + 4 mid + 8 outer
        _ring_config(r1=1.0, r2=0.4, r3=0.0, n1=8, n2=4, n3=1),
        # 1 center + 12 outer with slight phase offset on outer
        _ring_config(r1=1.0, r2=0.0, r3=0.0, n1=12, n2=0, n3=1, phase1=np.pi/12),
        # 2 center + 11 outer (approx)
        _ring_config(r1=1.0, r2=0.3, r3=0.0, n1=11, n2=2, n3=0, phase2=np.pi/2),
        # 3 center + 10 outer
        _ring_config(r1=1.0, r2=0.4, r3=0.0, n1=10, n2=3, n3=0, phase2=np.pi/3),
        # 1 center + 3 mid + 9 outer
        _ring_config(r1=1.0, r2=0.6, r3=0.0, n1=9, n2=3, n3=1),
        # 1 center + 2 mid + 10 outer
        _ring_config(r1=1.0, r2=0.7, r3=0.0, n1=10, n2=2, n3=1),
        # 1 center + 4 mid + 8 outer with different phase
        _ring_config(r1=1.0, r2=0.5, r3=0.0, n1=8, n2=4, n3=1, phase2=np.pi/4),
        # Additional diverse configurations
        _ring_config(r1=1.0, r2=0.35, r3=0.0, n1=9, n2=3, n3=1, phase2=np.pi/6),
        _ring_config(r1=1.0, r2=0.55, r3=0.0, n1=7, n2=5, n3=1, phase2=np.pi/3),
        _ring_config(r1=1.0, r2=0.45, r3=0.0, n1=10, n2=2, n3=1, phase2=np.pi/5),
        _ring_config(r1=1.0, r2=0.65, r3=0.0, n1=8, n2=4, n3=1, phase2=np.pi/7),
        _ring_config(r1=1.0, r2=0.5, r3=0.0, n1=11, n2=1, n3=1, phase2=np.pi/2),
    ]

    # Also include the original k_hull sweep as additional starts
    for k_hull in range(4, N):
        ring_starts.append(_make_config(k_hull, N - k_hull))

    # Allocate time per start: give more time to ring configs (first 15)
    n_starts = len(ring_starts)
    # Reserve ~15s for final polish
    search_budget = time_limit - 15.0
    per_start = search_budget / n_starts

    for idx, start_pts in enumerate(ring_starts):
        if time.time() > start_time + time_limit - 15:
            break
        
        # Give ring starts (first 15) more time
        if idx < 15:
            alloc = per_start * 1.5
        else:
            alloc = per_start * 0.5
            
        # Cap allocation to not exceed remaining time
        remaining = time_limit - 15 - (time.time() - start_time)
        alloc = min(alloc, remaining)
        
        if alloc < 2.0:
            break

        # Ensure at least 2.0 seconds per start
        alloc = max(alloc, 2.0)

        cand, cand_s = _optimize_from(start_pts, time.time() - start_time + alloc)
        if cand_s > best_score:
            best_score = cand_s
            best_points = cand.copy()

    # Final high-precision polish
    if best_points is not None and time.time() < start_time + time_limit - 12:
        # Stage 1: Direct local search for fine-grained improvements
        try:
            polished, pol_score = _local_search(best_points, start_time + time_limit - 8)
            if pol_score > best_score:
                best_score = pol_score
                best_points = polished.copy()
        except Exception:
            pass
        
        # Stage 2: High-beta soft-min L-BFGS-B
        if time.time() < start_time + time_limit - 8:
            try:
                result = opt.minimize(
                    lambda x: _soft_min_score(x.reshape(-1, 2), 500.0),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 80}
                )
                polished = result.x.reshape(-1, 2)
                pol_score, _ = compute_score(polished)
                if pol_score > best_score:
                    best_score = pol_score
                    best_points = polished.copy()
            except Exception:
                pass
        
        # Stage 3: Powell refinement
        if time.time() < start_time + time_limit - 5:
            try:
                res = opt.minimize(
                    _powell_objective,
                    best_points.flatten(),
                    method='Powell',
                    options={'maxiter': 200, 'ftol': 1e-13, 'xtol': 1e-11}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
            except Exception:
                pass
        
        # Stage 4: Final NM polish
        if time.time() < start_time + time_limit - 3:
            try:
                res = opt.minimize(
                    _hard_score_neg,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'adaptive': True, 'xatol': 1e-13, 'fatol': 1e-15}
                )
                final_points = res.x.reshape(-1, 2)
                final_score, _ = compute_score(final_points)
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
            except Exception:
                pass

    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)
    
    return best_points
