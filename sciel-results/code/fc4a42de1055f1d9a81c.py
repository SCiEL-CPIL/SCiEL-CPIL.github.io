"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import scipy.spatial
import time

N = 14

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s buffer before 120s limit

    # 1. Initialize with regular N-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.vstack([np.cos(angles), np.sin(angles)]).T.astype(np.float64)

    # 2. Efficient score function: min triangle area / convex hull area
    def get_score(pts):
        n = len(pts)
        # Compute all triangle areas efficiently
        # Area of triangle (i,j,k) = 0.5 * |cross(pts[j]-pts[i], pts[k]-pts[i])|
        min_area = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                v1 = pts[j] - pts[i]
                for k in range(j + 1, n):
                    v2 = pts[k] - pts[i]
                    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    if area < min_area:
                        min_area = area
        # Convex hull area
        hull = scipy.spatial.ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area < 1e-12:
            return 0.0
        return min_area / hull_area

    current_score = get_score(points)
    best_points = points.copy()
    best_score = current_score

    # 3. Local search: move points to optimize the ratio
    # Strategy: The objective is min_triangle_area / hull_area
    # We can improve by:
    #   a) Moving points inward to reduce hull area (if they're on the hull)
    #   b) Adjusting points to increase the minimum triangle area
    #   c) A combination via gradient-like search

    # Use a simple iterative local search with random perturbations
    # and coordinate descent on the ratio

    def optimize_local(pts, max_time=60.0):
        pts = pts.copy()
        current_score = get_score(pts)
        best = pts.copy()
        best_sc = current_score
        step_size = 0.05
        n_iter = 0
        t0 = time.time()

        while time.time() - t0 < max_time and time.time() < deadline:
            # Try random perturbations
            improved = False
            for _ in range(200):
                if time.time() > deadline:
                    break
                trial = pts.copy()
                idx = np.random.randint(0, N)
                direction = np.random.randn(2)
                direction /= (np.linalg.norm(direction) + 1e-15)
                trial[idx] += step_size * direction
                sc = get_score(trial)
                if sc > current_score + 1e-12:
                    pts = trial
                    current_score = sc
                    if sc > best_sc:
                        best = pts.copy()
                        best_sc = sc
                    improved = True
            if not improved:
                # Reduce step size
                step_size *= 0.7
                if step_size < 1e-6:
                    break
            n_iter += 1

        return best, best_sc

    # Run local search from the regular polygon
    best_points, best_score = optimize_local(points, max_time=50.0)

    # 4. Try multiple random restarts if time permits
    if time.time() < deadline - 20:
        n_restarts = 5
        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break
            # Random initialization near the regular polygon
            angles_r = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.random.randn(N) * 0.3
            pts_r = np.vstack([np.cos(angles_r), np.sin(angles_r)]).T
            # Scale to unit area roughly
            hull_r = scipy.spatial.ConvexHull(pts_r)
            pts_r = pts_r * np.sqrt(1.0 / hull_r.volume)

            res_pts, res_sc = optimize_local(pts_r, max_time=15.0)
            if res_sc > best_score:
                best_points = res_pts
                best_score = res_sc

    # 5. Final polish: finer local search
    if time.time() < deadline - 5:
        best_points, best_score = optimize_local(best_points, max_time=deadline - time.time() - 3)

    return best_points
# EVOLVE-BLOCK-END
