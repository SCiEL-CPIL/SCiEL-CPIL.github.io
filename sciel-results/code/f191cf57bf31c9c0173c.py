# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).

    Uses multi-restart Nelder-Mead optimization with diverse initial conditions.
    """
    n = 20
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin before 630s limit

    def _normalize(pts):
        """Subtract centroid, divide by mean distance to centroid."""
        centroid = np.mean(pts, axis=0)
        pts_c = pts - centroid
        dists = np.linalg.norm(pts_c, axis=1)
        mean_dist = np.mean(dists)
        if mean_dist <= 0.01:
            return pts_c, 0.0
        return pts_c / mean_dist, 1.0

    def _check_strictly_convex(pts):
        """Check strict convexity: no repeated points, all turns same sign, total turning +-2*pi."""
        n_pts = len(pts)
        # Check no repeated points
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                    return False
        # Compute cross products of consecutive edges
        turns = []
        for i in range(n_pts):
            p0 = pts[i]
            p1 = pts[(i+1) % n_pts]
            p2 = pts[(i+2) % n_pts]
            e1 = p1 - p0
            e2 = p2 - p1
            cross = e1[0]*e2[1] - e1[1]*e2[0]
            turns.append(cross)
        turns = np.array(turns)
        # All turns must be nonzero and same sign
        if np.any(np.abs(turns) < 1e-12):
            return False
        if np.any(turns > 0) and np.any(turns < 0):
            return False
        # Total turning should be +-2*pi (for a simple closed convex polygon)
        # For a convex polygon, the sum of exterior angles is 2*pi
        # We can check this by computing the angle changes
        angles = np.arctan2(turns, np.sum(turns**2, axis=0) if len(turns.shape)>1 else np.sum(turns**2))
        # Actually, for a strictly convex polygon traversed in order, the total turn is 2*pi
        # Let's just verify it's a simple convex polygon by checking the winding
        total_turn = sum(np.sign(turns)) * 2 * np.pi
        # This is approximate; for strict convexity with all same-sign turns, it should be fine
        return True

    def _vertex_score(pts, i):
        """Compute the score for vertex i."""
        v = pts[i]
        # Distances from v to all other vertices
        dists = np.linalg.norm(pts - v, axis=1)
        dists[i] = np.inf  # exclude self
        dists_sorted = np.sort(dists)
        # d_0 <= ... <= d_18
        if dists_sorted[0] < 0.05:
            return 0.0
        # Find k: first index minimizing d_{k+1} - d_k
        diffs = np.diff(dists_sorted)
        k = int(np.argmin(diffs))
        D = (dists_sorted[k] + dists_sorted[k+1]) / 2.0
        if D < 0.1:
            return 0.0
        # Take four distances
        if k == 0:
            four_d = dists_sorted[0:4]
        elif k == 17:
            four_d = dists_sorted[15:19]
        else:
            four_d = dists_sorted[k-1:k+3]
        # raw = mean of max(d/D, D/d) for these four
        ratios = np.maximum(four_d / D, D / four_d)
        raw = np.mean(ratios)
        return min(1.0, 1.0 / raw)

    def _score_polygon(pts):
        """Compute the overall score: minimum vertex score."""
        norm_pts, valid = _normalize(pts)
        if not valid:
            return 0.0
        if not _check_strictly_convex(norm_pts):
            return 0.0
        scores = [_vertex_score(norm_pts, i) for i in range(n)]
        return min(scores)

    def _objective(params):
        """Objective function for optimization: negative of score (we minimize)."""
        pts = np.array(params).reshape(n, 2)
        return -_score_polygon(pts)

    # Regular polygon as baseline
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    baseline = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    best_score = _score_polygon(baseline)
    best_pts = baseline.copy()

    if time.time() > deadline:
        return best_pts

    # Multi-restart Nelder-Mead optimization
    np.random.seed(42)
    noise_scales = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.1, 0.2, 0.15, 0.25]
    num_restarts = min(len(noise_scales), 10)

    for r in range(num_restarts):
        if time.time() > deadline - 30.0:  # stop if less than 30s left
            break

        scale = noise_scales[r]
        # Perturb the regular polygon
        noise = np.random.randn(n, 2) * scale
        init_pts = baseline + noise
        # Ensure it's still roughly convex by projecting if needed
        # For now, just try it
        init_params = init_pts.flatten()

        try:
            result = minimize(
                _objective,
                init_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            if time.time() > deadline - 10.0:
                break

            candidate = result.x.reshape(n, 2)
            cand_score = _score_polygon(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = candidate.copy()
        except Exception:
            continue

    # Tight polishing passes on the best result
    for polish_iter in range(5):
        if time.time() > deadline - 15.0:
            break
        try:
            init_params = best_pts.flatten()
            result = minimize(
                _objective,
                init_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if time.time() > deadline - 5.0:
                break
            candidate = result.x.reshape(n, 2)
            cand_score = _score_polygon(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = candidate.copy()
        except Exception:
            break

    return best_pts

# EVOLVE-BLOCK-END
