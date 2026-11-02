import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety

    # Precompute indices for all triples as numpy arrays for vectorized computation
    tri_i = []
    tri_j = []
    tri_k = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                tri_i.append(i)
                tri_j.append(j)
                tri_k.append(k)
    tri_i = np.array(tri_i, dtype=np.int32)
    tri_j = np.array(tri_j, dtype=np.int32)
    tri_k = np.array(tri_k, dtype=np.int32)
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
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        score = min_area / area_hull
        return score, area_hull

    def _soft_min_score(points, beta=50.0):
        """Smooth surrogate: log-sum-exp approximation of the minimum triangle area."""
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 1e6
        if area_hull < 1e-9:
            return 1e6
        x = points[:, 0]
        y = points[:, 1]
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
        score = soft_min_area / area_hull
        return -score

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    # ---- Ring-structure machinery (from inspiration) ----
    # A ring structure is a list of (count, radius, phase) per ring.
    # The optimization searches only 2*len(rings) parameters (radius, phase per ring),
    # not 2*N free coordinates. This is the structural reduction.
    def _build_from_rings(ring_counts, params):
        """Build N points from ring counts and (radius, phase) params per ring."""
        pts = []
        for i, count in enumerate(ring_counts):
            if count == 0:
                continue
            radius = abs(params[2 * i])
            phase = params[2 * i + 1]
            if count == 1:
                pts.append(np.array([[radius * np.cos(phase), radius * np.sin(phase)]]))
            else:
                angles = phase + 2 * np.pi * np.arange(count) / count
                ring_pts = radius * np.column_stack([np.cos(angles), np.sin(angles)])
                pts.append(ring_pts)
        if not pts:
            return np.zeros((N, 2))
        return np.vstack(pts)

    def _ring_objective(params, ring_counts, beta):
        """Soft-min objective over ring parameters (for gradient-based optimization)."""
        pts = _build_from_rings(ring_counts, params)
        return _soft_min_score(pts, beta)

    def _ring_objective_hard(params, ring_counts):
        """Hard (true) negative score over ring parameters."""
        pts = _build_from_rings(ring_counts, params)
        s, _ = compute_score(pts)
        return -s

    def _optimize_ring(ring_counts, x0, beta, max_iter=200, method='L-BFGS-B'):
        """Run one optimization pass on ring parameters. Returns (params, score)."""
        try:
            if method == 'L-BFGS-B':
                result = opt.minimize(
                    lambda p: _ring_objective(p, ring_counts, beta),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 80}
                )
            else:
                result = opt.minimize(
                    lambda p: _ring_objective_hard(p, ring_counts),
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-14}
                )
            params = result.x
            pts = _build_from_rings(ring_counts, params)
            s, _ = compute_score(pts)
            return params, s, pts
        except Exception:
            return x0, 0.0, None

    best_points = None
    best_score = -float('inf')

    # ---- Phase 1: Explore ring structures (reduced-dimension search) ----
    # Ring structures summing to 13. The outer ring is typically the convex hull.
    ring_structures = [
        [13],          # single 13-gon
        [1, 12],       # center + 12-gon
        [1, 6, 6],     # center + two 6-gons
        [7, 6],        # 7-gon + 6-gon
        [4, 9],        # 4-gon + 9-gon
        [5, 8],        # 5-gon + 8-gon
        [6, 7],        # 6-gon + 7-gon
        [2, 11],       # 2 + 11
        [3, 10],       # 3 + 10
    ]

    rng = np.random.RandomState(42)
    betas_ring = [40.0, 150.0, 500.0, 1200.0]

    for ring_counts in ring_structures:
        if time.time() > start_time + time_limit - 35:
            break
        n_rings = len(ring_counts)
        # Multiple initializations per structure
        n_init = 3
        for init_i in range(n_init):
            if time.time() > start_time + time_limit - 30:
                break
            # Initial radii: decreasing, with slight randomness
            x0 = np.zeros(2 * n_rings)
            for i in range(n_rings):
                x0[2 * i] = 1.0 / (i + 1)  # decreasing radii
                x0[2 * i + 1] = 0.0
            if init_i > 0:
                # Perturb
                x0 += rng.randn(2 * n_rings) * 0.15
                x0[0::2] = np.abs(x0[0::2])  # keep radii non-negative

            # Beta annealing on ring params
            current_params = x0
            for beta in betas_ring:
                if time.time() > start_time + time_limit - 25:
                    break
                params, s, pts = _optimize_ring(ring_counts, current_params, beta, max_iter=120, method='L-BFGS-B')
                if s > best_score and pts is not None:
                    best_score = s
                    best_points = pts.copy()
                current_params = params

            # NM polish on ring params
            if time.time() < start_time + time_limit - 20:
                params, s, pts = _optimize_ring(ring_counts, current_params, 0, max_iter=150, method='Nelder-Mead')
                if s > best_score and pts is not None:
                    best_score = s
                    best_points = pts.copy()

    # ---- Phase 2: Free-coordinate refinement on best ring solution ----
    # Now use the full 26-dim space to break symmetry and fine-tune.
    if best_points is not None and time.time() < start_time + time_limit - 30:
        current = best_points.copy()
        betas_free = [100.0, 400.0, 1000.0]
        for beta in betas_free:
            if time.time() > start_time + time_limit - 20:
                break
            try:
                result = opt.minimize(
                    lambda x: _soft_min_score(x.reshape(-1, 2), beta),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100}
                )
                optimized = result.x.reshape(-1, 2)
                s, _ = compute_score(optimized)
                if s > best_score:
                    best_score = s
                    best_points = optimized.copy()
                current = optimized
            except Exception:
                pass

    # ---- Phase 3: NM polish in full space ----
    if best_points is not None and time.time() < start_time + time_limit - 15:
        try:
            res = opt.minimize(
                lambda x: -compute_score(x.reshape(-1, 2))[0],
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'adaptive': True, 'xatol': 1e-13, 'fatol': 1e-15}
            )
            nm_points = res.x.reshape(-1, 2)
            s, _ = compute_score(nm_points)
            if s > best_score:
                best_score = s
                best_points = nm_points.copy()
        except Exception:
            pass

    # ---- Phase 4: Final high-beta L-BFGS polish ----
    if best_points is not None and time.time() < start_time + time_limit - 8:
        try:
            result = opt.minimize(
                lambda x: _soft_min_score(x.reshape(-1, 2), 2000.0),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 100}
            )
            polished = result.x.reshape(-1, 2)
            s, _ = compute_score(polished)
            if s > best_score:
                best_score = s
                best_points = polished.copy()
        except Exception:
            pass

    # ---- Phase 5: Final NM ----
    if best_points is not None and time.time() < start_time + time_limit - 3:
        try:
            res = opt.minimize(
                lambda x: -compute_score(x.reshape(-1, 2))[0],
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 400, 'adaptive': True, 'xatol': 1e-14, 'fatol': 1e-16}
            )
            final_points = res.x.reshape(-1, 2)
            s, _ = compute_score(final_points)
            if s > best_score:
                best_score = s
                best_points = final_points.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)

    return best_points

# EVOLVE-BLOCK-END
