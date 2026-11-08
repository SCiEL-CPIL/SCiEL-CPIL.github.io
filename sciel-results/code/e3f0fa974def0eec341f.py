import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 140.0

    # Precompute triangle indices for vectorization
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)

    def compute_score(points):
        """
        Compute min triangle area / convex hull area.
        Returns 0.0 if invalid.
        """
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
        
        if min_area < 0:
            return 0.0
        return min_area / area_hull

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def points_from_radii(radii):
        """
        Generate points from 13 radii with fixed angles 2*pi*k/13.
        """
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        return np.column_stack([xs, ys])

    def objective_symmetric(radii):
        """
        Objective function for symmetric optimization.
        Returns negative score to be minimized.
        """
        pts = points_from_radii(radii)
        score = compute_score(pts)
        return -score

    def objective_full(points_flat):
        """
        Objective function for full 26D optimization.
        Returns negative score to be minimized.
        """
        pts = points_flat.reshape(-1, 2)
        score = compute_score(pts)
        return -score

    best_score = -float('inf')
    best_points = None

    def update_best(pts):
        nonlocal best_score, best_points
        score = compute_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
        return score

    # Phase 1: Symmetric Subspace Optimization (High quality baseline)
    # This approach is robust and often finds the global optimum for this type of problem
    initializations_sym = [
        np.ones(N),
        np.ones(N) + 0.05 * np.sin(2 * np.pi * np.arange(N) / N),
        np.ones(N) + 0.05 * np.sin(4 * np.pi * np.arange(N) / N),
    ]

    for init_rad in initializations_sym:
        if time.time() > start_time + time_limit - 20:
            break
        try:
            res = opt.minimize(
                objective_symmetric,
                init_rad,
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            sym_pts = points_from_radii(res.x)
            update_best(sym_pts)
        except Exception:
            continue

    # Phase 2: Full Space Polish
    # Start from the best symmetric configuration found
    if best_points is not None:
        if time.time() > start_time + time_limit - 10:
            return best_points
            
        try:
            # Nelder-Mead is good for this size
            res = opt.minimize(
                objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            candidate_pts = res.x.reshape(-1, 2)
            update_best(candidate_pts)
        except Exception:
            pass

    # Phase 3: Diverse Topology Search
    # Try other structures if time permits
    rng_global = np.random.RandomState(42)
    
    # Candidate 1: Regular Polygon
    if time.time() < start_time + time_limit - 15:
        update_best(_regular_polygon(N, 1.0))

    # Candidate 2: 12 Outer + 1 Center
    if time.time() < start_time + time_limit - 15:
        pts_12_1 = np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])])
        update_best(pts_12_1)

    # Candidate 3: Random Points in Disk
    for _ in range(5):
        if time.time() > start_time + time_limit - 5:
            break
        # Random points in disk
        r = np.sqrt(rng_global.uniform(0, 1, N))
        theta = rng_global.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        
        # Quick polish
        if time.time() > start_time + time_limit - 5:
            continue
        try:
            res = opt.minimize(
                objective_full,
                pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            candidate_pts = res.x.reshape(-1, 2)
            update_best(candidate_pts)
        except Exception:
            continue

    if best_points is None:
        best_points = _regular_polygon(N, 1.0)

    return best_points
