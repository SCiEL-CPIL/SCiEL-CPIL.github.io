import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
N_HULL = 12
N_INNER = 1

def construct():
    start_time = time.time()
    time_limit = 142.0  # Safety margin

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0

        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _soft_objective(points_flat, beta=200.0):
        """Objective: negative soft-min triangle area (normalized)."""
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)

        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross

        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        return -soft_min_area

    def _exact_objective(points_flat):
        """Objective: negative min triangle area (normalized)."""
        pts = points_flat.reshape(-1, 2)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    # ---- Radial Fourier parameterization for the 12 hull points ----
    # Hull vertices at angles theta_k = 2*pi*k/12, radius r(theta) = 1 + sum a_m cos(m*theta) + b_m sin(m*theta)
    # Inner point at (cx, cy).
    # We use harmonics m = 1..M. Due to D12 symmetry we could restrict, but for flexibility we allow all.
    # To preserve convexity, keep amplitudes small.

    def _fourier_points(params):
        """
        params: array of shape (2*M + 2,)
        First 2*M entries: [a_1, b_1, a_2, b_2, ..., a_M, b_M]
        Last 2 entries: [cx, cy] for the inner point.
        Returns array of shape (13, 2).
        """
        M = (len(params) - 2) // 2
        thetas = 2 * np.pi * np.arange(N_HULL) / N_HULL
        radii = np.ones(N_HULL)
        for m in range(1, M + 1):
            a = params[2 * (m - 1)]
            b = params[2 * (m - 1) + 1]
            radii += a * np.cos(m * thetas) + b * np.sin(m * thetas)
        hull_xs = radii * np.cos(thetas)
        hull_ys = radii * np.sin(thetas)
        cx = params[-2]
        cy = params[-1]
        return np.column_stack([np.append(hull_xs, cx), np.append(hull_ys, cy)])

    def _fourier_objective(params_flat, beta=200.0):
        pts = _fourier_points(params_flat)
        return _soft_objective(pts.flatten(), beta=beta)

    def _fourier_exact_objective(params_flat):
        pts = _fourier_points(params_flat)
        return _exact_objective(pts.flatten())

    def _powell_polish(points, max_iter=300):
        """Powell method for robust local optimization."""
        pts = points.copy()
        try:
            res = opt.minimize(
                _exact_objective,
                pts.flatten(),
                method='Powell',
                options={'maxiter': max_iter, 'xtol': 1e-14, 'ftol': 1e-14}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            return candidate, score
        except Exception:
            return pts, compute_score(pts)

    def _nelder_mead_polish(points, max_iter=300):
        """Nelder-Mead for final polish."""
        pts = points.copy()
        try:
            res = opt.minimize(
                _exact_objective,
                pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            return candidate, score
        except Exception:
            return pts, compute_score(pts)

    def _coordinate_descent(points, max_iter=300, tol=1e-12):
        """Coordinate descent with adaptive steps."""
        pts = points.copy()
        best_score = compute_score(pts)
        base_steps = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]

        for iteration in range(max_iter):
            if time.time() > start_time + time_limit - 2:
                break
            improved = False
            for i in range(N):
                best_local = pts[i].copy()
                best_local_score = best_score
                for step in base_steps:
                    directions = [
                        (step, 0), (-step, 0), (0, step), (0, -step),
                        (step * 0.707, step * 0.707), (-step * 0.707, step * 0.707),
                        (step * 0.707, -step * 0.707), (-step * 0.707, -step * 0.707)
                    ]
                    for dx, dy in directions:
                        new_pt = pts.copy()
                        new_pt[i] = pts[i] + np.array([dx, dy])
                        score = compute_score(new_pt)
                        if score > best_local_score + tol:
                            best_local_score = score
                            best_local = new_pt[i].copy()
                            improved = True
                if improved:
                    pts[i] = best_local
                    best_score = best_local_score
            if not improved:
                break
        return pts, best_score

    # ---- Initializations ----
    # Primary: regular 12-gon + center
    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    initializations = []

    # 1. Regular 12-gon + center
    init1 = np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])])
    initializations.append(init1)

    # 2. Regular 13-gon
    initializations.append(_regular_polygon(13, 1.0))

    # 3. Regular 12-gon + off-center inner point
    for r in [0.1, 0.2, 0.3]:
        for angle in [0, np.pi / 6, np.pi / 3]:
            inner = np.array([[r * np.cos(angle), r * np.sin(angle)]])
            initializations.append(np.vstack([_regular_polygon(12, 1.0), inner]))

    # 4. Slightly deformed 12-gon + center
    rng = np.random.RandomState(42)
    for trial in range(5):
        deformed = _regular_polygon(12, 1.0)
        deformed += rng.randn(12, 2) * 0.05
        initializations.append(np.vstack([deformed, np.array([[0.0, 0.0]])]))

    # 5. Regular 11-gon + 2 inner
    for r in [0.2, 0.4, 0.6]:
        for angle in [0, np.pi]:
            inner1 = np.array([[r * np.cos(angle), r * np.sin(angle)]])
            inner2 = np.array([[r * np.cos(angle + np.pi), r * np.sin(angle + np.pi)]])
            initializations.append(np.vstack([_regular_polygon(11, 1.0), inner1, inner2]))

    # 6. Regular 10-gon + 3 inner (triangle)
    for r in [0.2, 0.4, 0.6]:
        initializations.append(np.vstack([_regular_polygon(10, 1.0), _regular_polygon(3, r)]))

    # 7. 9 outer + 4 inner
    for r in [0.2, 0.4, 0.6]:
        initializations.append(np.vstack([_regular_polygon(9, 1.0), _regular_polygon(4, r)]))

    # 8. 8 outer + 5 inner
    for r in [0.2, 0.4, 0.6]:
        initializations.append(np.vstack([_regular_polygon(8, 1.0), _regular_polygon(5, r)]))

    # 9. 7 outer + 6 inner
    for r in [0.2, 0.4, 0.6]:
        initializations.append(np.vstack([_regular_polygon(7, 1.0), _regular_polygon(6, r)]))

    # 10. 6 outer + center + 6 inner
    for r in [0.2, 0.4, 0.6]:
        initializations.append(np.vstack([_regular_polygon(6, 1.0), np.array([[0.0, 0.0]]), _regular_polygon(6, r)]))

    # 11. 5 outer + 8 inner
    for r in [0.2, 0.4, 0.6]:
        initializations.append(np.vstack([_regular_polygon(5, 1.0), _regular_polygon(8, r)]))

    # 12. 4 outer + 9 inner
    for r in [0.2, 0.4, 0.6]:
        initializations.append(np.vstack([_regular_polygon(4, 1.0), _regular_polygon(9, r)]))

    best_score = -float('inf')
    best_points = None

    # ---- Fourier-based global search ----
    # Use 2 harmonics (m=1,2) -> 4 params + 2 for center = 6 params
    M = 2
    n_fourier_params = 2 * M + 2
    bounds_fourier = [(-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15), (-0.5, 0.5), (-0.5, 0.5)]

    if time.time() < start_time + time_limit - 25:
        try:
            de_res = opt.differential_evolution(
                lambda x: _fourier_objective(x, beta=100.0),
                bounds_fourier,
                seed=42,
                maxiter=40,
                tol=1e-6,
                polish=True
            )
            if de_res.success:
                fourier_pts = _fourier_points(de_res.x)
                score_fourier = compute_score(fourier_pts)
                if score_fourier > best_score:
                    best_score = score_fourier
                    best_points = fourier_pts.copy()
                initializations.append(fourier_pts)
        except Exception:
            pass

    # Try with 3 harmonics as well
    if time.time() < start_time + time_limit - 15:
        M2 = 3
        n_fourier_params2 = 2 * M2 + 2
        bounds_fourier2 = [(-0.1, 0.1)] * 6 + [(-0.5, 0.5), (-0.5, 0.5)]
        try:
            de_res2 = opt.differential_evolution(
                lambda x: _fourier_objective(x, beta=100.0),
                bounds_fourier2,
                seed=123,
                maxiter=30,
                tol=1e-6,
                polish=True
            )
            if de_res2.success:
                fourier_pts2 = _fourier_points(de_res2.x)
                score_fourier2 = compute_score(fourier_pts2)
                if score_fourier2 > best_score:
                    best_score = score_fourier2
                    best_points = fourier_pts2.copy()
                initializations.append(fourier_pts2)
        except Exception:
            pass

    # ---- Optimize each initialization ----
    for init in initializations:
        if time.time() > start_time + time_limit - 8:
            break

        current = init.copy()

        # Step 1: L-BFGS-B with soft objective (annealing beta)
        betas = [20.0, 50.0, 100.0, 200.0, 500.0]
        for beta in betas:
            if time.time() > start_time + time_limit - 4:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 60, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass

        # Step 2: Coordinate descent refinement
        if time.time() > start_time + time_limit - 3:
            break
        try:
            current, score_cd = _coordinate_descent(current, max_iter=200)
            if score_cd > best_score:
                best_score = score_cd
                best_points = current.copy()
        except Exception:
            pass

        # Step 3: Powell method
        if time.time() > start_time + time_limit - 2:
            break
        try:
            candidate, score_powell = _powell_polish(current, max_iter=150)
            if score_powell > best_score:
                best_score = score_powell
                best_points = candidate.copy()
            current = candidate
        except Exception:
            pass

        # Step 4: Nelder-Mead final polish
        if time.time() > start_time + time_limit - 1:
            break
        try:
            candidate, score_nm = _nelder_mead_polish(current, max_iter=200)
            if score_nm > best_score:
                best_score = score_nm
                best_points = candidate.copy()
        except Exception:
            pass

        # Step 5: Additional coordinate descent
        if time.time() > start_time + time_limit - 0.5:
            break
        try:
            candidate, score_cd2 = _coordinate_descent(candidate, max_iter=100)
            if score_cd2 > best_score:
                best_score = score_cd2
                best_points = candidate.copy()
        except Exception:
            pass

    # ---- Final high-precision polish on the best point found ----
    if best_points is not None and time.time() < start_time + time_limit - 2:
        try:
            # Very tight Nelder-Mead
            res_final = opt.minimize(
                _exact_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-16}
            )
            candidate_final = res_final.x.reshape(-1, 2)
            score_final = compute_score(candidate_final)
            if score_final > best_score:
                best_score = score_final
                best_points = candidate_final.copy()
        except Exception:
            pass

    # Final coordinate descent with very small steps
    if best_points is not None and time.time() < start_time + time_limit - 0.5:
        try:
            candidate_final2, score_final2 = _coordinate_descent(best_points, max_iter=50, tol=1e-14)
            if score_final2 > best_score:
                best_score = score_final2
                best_points = candidate_final2.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = initializations[0]

    return best_points
