# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _vertex_score(distances):
    """Compute the per-vertex score from sorted distances to other vertices."""
    d = np.sort(distances)
    # d has 19 elements
    if np.any(d < 0.05):
        return 0.0
    gaps = np.diff(d)
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        idx = [0, 1, 2, 3]
    elif k == 17:
        idx = [15, 16, 17, 18]
    else:
        idx = [k - 1, k, k + 1, k + 2]
    four_d = d[idx]
    raw = np.mean(np.maximum(four_d / D, D / four_d))
    return min(1.0, 1.0 / raw)


def _polygon_score(polygon):
    """Compute the overall score: min vertex score over all vertices."""
    n = len(polygon)
    scores = []
    for i in range(n):
        v = polygon[i]
        dists = np.sqrt(np.sum((polygon - v) ** 2, axis=1))
        # Exclude self-distance
        dists = np.delete(dists, i)
        scores.append(_vertex_score(dists))
    return min(scores)


def _is_valid_convex(polygon):
    """Check if polygon is strictly convex with no repeated points."""
    n = len(polygon)
    # Check no repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.allclose(polygon[i], polygon[j], atol=1e-10):
                return False
    # Check all turns have same sign and are nonzero
    signs = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if abs(cross) < 1e-12:
            return False
        signs.append(np.sign(cross))
    if not all(s == signs[0] for s in signs):
        return False
    return True


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).

    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    n = 20
    deadline = time.time() + 580  # Safety margin before 630s limit
    
    # Start from regular polygon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    best_score = _polygon_score(best_polygon)
    
    # Try to improve using optimization
    # Parameterize by angles (radii fixed at 1 for simplicity, or optimize both)
    
    def _score_from_params(params):
        """Convert parameters to polygon and compute score. Return negative for minimization."""
        # params: n angles (or we can use n radii + n angles)
        # Let's use angles only, radius = 1
        a = params
        # Normalize angles to [0, 2*pi) and sort
        a = np.sort(a % (2 * np.pi))
        # Ensure they are in [0, 2*pi)
        x = np.cos(a)
        y = np.sin(a)
        poly = np.stack([x, y], axis=1)
        if not _is_valid_convex(poly):
            return 0.0
        return _polygon_score(poly)
    
    # Try Nelder-Mead optimization on angles
    x0 = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def neg_score(params):
        s = _score_from_params(params)
        return -s
    
    # Use multiple random restarts
    rng = np.random.default_rng(42)
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        if r == 0:
            x_start = x0.copy()
        else:
            # Perturb the regular polygon angles
            x_start = x0 + rng.normal(0, 0.05, n)
        
        try:
            result = minimize(neg_score, x_start, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if result.fun < 0:
                s = -result.fun
                if s > best_score:
                    a = np.sort(result.x % (2 * np.pi))
                    poly = np.stack([np.cos(a), np.sin(a)], axis=1)
                    if _is_valid_convex(poly):
                        best_score = s
                        best_polygon = poly
        except Exception:
            pass
    
    return best_polygon


# EVOLVE-BLOCK-END
