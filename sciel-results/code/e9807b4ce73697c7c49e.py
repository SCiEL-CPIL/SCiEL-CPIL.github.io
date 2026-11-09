"""Heilbronn convex regions: 13 points, maximize min triangle area / hull area."""
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

N = 13
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # return before this margin


def _min_triangle_area(pts):
    """Compute the minimum area over all triples of points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            for k in range(j + 1, n):
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use polygon area via shoelace
        hull_idx = ConvexHull(pts).vertices
        hp = pts[hull_idx]
        n = len(hp)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += hp[i][0] * hp[j][1] - hp[j][0] * hp[i][1]
        return abs(area) * 0.5


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area


def _initial_guess():
    """Regular 13-gon on unit circle."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _local_search(pts, time_limit, rng):
    """Local search: perturb points, keep if score improves."""
    best_pts = pts.copy()
    best_score = _score(best_pts)
    step_size = 0.05
    start = time.time()
    
    while time.time() - start < time_limit:
        # Try random perturbations
        improved = False
        for _ in range(50):
            if time.time() - start > time_limit:
                break
            new_pts = best_pts.copy()
            idx = rng.randint(0, N - 1)
            noise = np.random.normal(0, step_size, 2)
            new_pts[idx] += noise
            new_score = _score(new_pts)
            if new_score > best_score:
                best_pts = new_pts
                best_score = new_score
                improved = True
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return best_pts, best_score


def _delaunay_guided_search(pts, time_limit, rng):
    """Use Delaunay triangulation to identify small triangles and fix them."""
    best_pts = pts.copy()
    best_score = _score(best_pts)
    start = time.time()
    
    while time.time() - start < time_limit:
        try:
            tri = Delaunay(best_pts)
        except Exception:
            break
        
        # Find the smallest triangle in the Delaunay triangulation
        min_area = float('inf')
        min_tri = None
        for simplex in tri.simplices:
            i, j, k = simplex
            v1 = best_pts[j] - best_pts[i]
            v2 = best_pts[k] - best_pts[i]
            area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
            if area < min_area:
                min_area = area
                min_tri = (i, j, k)
        
        if min_tri is None:
            break
        
        # Try to increase the area of the smallest triangle by moving one of its vertices
        i, j, k = min_tri
        improved = False
        for idx in [i, j, k]:
            for _ in range(20):
                if time.time() - start > time_limit:
                    break
                new_pts = best_pts.copy()
                # Move the point outward from the centroid of the triangle
                centroid = (best_pts[i] + best_pts[j] + best_pts[k]) / 3
                direction = best_pts[idx] - centroid
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm
                else:
                    direction = np.random.normal(0, 1, 2)
                new_pts[idx] += 0.02 * direction
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_pts = new_pts
                    best_score = new_score
                    improved = True
        
        if not improved:
            break
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    rng = np.random.RandomState(N)
    
    # Start with regular 13-gon
    best_pts = _initial_guess()
    best_score = _score(best_pts)
    
    # Phase 1: Local search
    remaining = DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
    if remaining > 5:
        best_pts, best_score = _local_search(best_pts, min(remaining, 30), rng)
    
    # Phase 2: Delaunay-guided search
    remaining = DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
    if remaining > 5:
        best_pts, best_score = _delaunay_guided_search(best_pts, min(remaining, 60), rng)
    
    # Phase 3: Final local search with smaller steps
    remaining = DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
    if remaining > 5:
        best_pts, best_score = _local_search(best_pts, remaining, rng)
    
    # Final check: ensure we return within deadline
    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        # Return whatever we have
        pass
    
    return best_pts
