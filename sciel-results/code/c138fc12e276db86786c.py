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
    n_tris = len(tri_i)

    def compute_score(points):
        """
        Compute min triangle area / convex hull area.
        Returns 0.0 if invalid.
        """
        try:
            pts = points - points.mean(axis=0)
            # Check for degenerate hull
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

    def points_from_fourier(coeffs, max_k=3):
        """
        Generate points from Fourier coefficients a_k, b_k for k=1..max_k.
        r(theta) = 1 + sum_{k=1}^{max_k} (a_k cos(k theta) + b_k sin(k theta))
        Points are placed at angles theta_j = 2*pi*j/N.
        """
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N)
        for k in range(1, max_k + 1):
            a_k = coeffs[2 * (k - 1)]
            b_k = coeffs[2 * (k - 1) + 1]
            radii += a_k * np.cos(k * angles) + b_k * np.sin(k * angles)
        
        # Ensure radii are positive to maintain convexity/star-shape from origin
        # If any radius is <= 0, the point is at the origin or inside, which might be valid but let's clamp
        # Actually, if r < 0, the point is in the opposite direction. For convexity relative to origin, we want r > 0.
        # However, the problem asks for a convex region. A star-shaped function with r>0 is not necessarily convex.
        # But small perturbations around 1 keep it close to a circle, which is convex.
        # Let's just compute the points.
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

    def objective_fourier(coeffs):
        """
        Objective function for Fourier optimization.
        """
        pts = points_from_fourier(coeffs)
        score = compute_score(pts)
        return -score

    def objective_full(points_flat):
        """
        Objective function for full 26D optimization.
        """
        pts = points_flat.reshape(-1, 2)
        score = compute_score(pts)
        return -score

    best_score = -float('inf')
    best_points = None

    def try_update(pts):
        nonlocal best_score, best_points
        score = compute_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Phase 1: Optimize in the symmetric subspace (13 radii)
    # This is a strong baseline.
    initializations = [
        np.ones(N),
        np.ones(N) + 0.1 * np.sin(2 * np.pi * np.arange(N) / N),
        np.ones(N) + 0.1 * np.sin(4 * np.pi * np.arange(N) / N),
    ]

    for init_rad in initializations:
        if time.time() > start_time + time_limit - 20:
            break
        try:
            res = opt.minimize(
                objective_symmetric,
                init_rad,
                method='Powell',
                options={'maxiter': 300, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            candidate_pts = points_from_radii(res.x)
            try_update(candidate_pts)
        except Exception:
            continue

    # Phase 2: Optimize in the Fourier subspace (small dimension, e.g., 6 coefficients for k=1,2,3)
    # This allows for shape variations that are not strictly radial in the 13-point sense,
    # but defined on a continuous curve sampled at 13 points.
    # Actually, points_from_fourier uses fixed angles. So it's a subset of the 13-radii space?
    # No. The 13-radii space allows any radius at each fixed angle.
    # The Fourier space with max_k=3 imposes a smoothness constraint on the radii.
    # This might help find smoother, potentially better shapes if the optimum is smooth.
    
    # Let's try max_k=2 and max_k=3
    for max_k in [2, 3]:
        dim = 2 * max_k
        if time.time() > start_time + time_limit - 15:
            break
        initial_coeffs = np.zeros(dim)
        try:
            res = opt.minimize(
                objective_fourier,
                initial_coeffs,
                method='Powell',
                options={'maxiter': 300, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            candidate_pts = points_from_fourier(res.x, max_k)
            try_update(candidate_pts)
        except Exception:
            continue

    # Phase 3: Polish in the full 26-dimensional space
    # Start from the best configuration found so far
    if best_points is not None:
        if time.time() > start_time + time_limit - 10:
            return best_points
            
        try:
            res = opt.minimize(
                objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            candidate_pts = res.x.reshape(-1, 2)
            try_update(candidate_pts)
        except Exception:
            pass

    # Phase 4: Random restarts in full space
    if time.time() < start_time + time_limit - 10:
        rng = np.random.RandomState(42)
        for _ in range(3):
            if time.time() > start_time + time_limit - 5:
                break
            
            # Random points in a circle, sorted by angle to maintain some structure
            r = rng.uniform(0.5, 1.0, N)
            theta = rng.uniform(0, 2 * np.pi, N)
            theta.sort()
            pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            
            try:
                res = opt.minimize(
                    objective_full,
                    pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                candidate_pts = res.x.reshape(-1, 2)
                try_update(candidate_pts)
            except Exception:
                continue

    if best_points is None:
        best_points = _regular_polygon(N, 1.0)

    return best_points
