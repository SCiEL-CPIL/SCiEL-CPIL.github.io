"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import itertools
import time


def _min_triangle_area(pts):
    """Compute the minimum triangle area over all triples of points."""
    n = len(pts)
    min_area = float('inf')
    # Use vectorized computation for speed
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs(
                    (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) -
                    (pts[k, 0] - pts[i, 0]) * (pts[j, 1] - pts[i, 1])
                )
                if area < min_area:
                    min_area = area
    return min_area


def _score(pts):
    """Score = min triangle area / convex hull area."""
    try:
        if len(pts) < 3:
            return 0.0
        # Check for collinear or degenerate
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, .volume gives the area
        if hull_area <= 0:
            return 0.0
        min_area = _min_triangle_area(pts)
        return min_area / hull_area
    except Exception:
        return 0.0


def construct():
    deadline = time.time() + 110  # 10s safety margin before 120s limit
    
    # Start with regular polygon as a good baseline
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _score(best_pts)
    
    # Try several angular offsets and scales
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        offset = np.random.uniform(0, 2 * np.pi)
        angles_test = angles + offset
        pts_test = np.column_stack((np.cos(angles_test), np.sin(angles_test)))
        s = _score(pts_test)
        if s > best_score:
            best_score = s
            best_pts = pts_test.copy()
    
    # Local optimization: perturb each point slightly and keep improvements
    # Use coordinate descent with small random perturbations
    for iteration in range(200):
        if time.time() > deadline - 10:
            break
        
        improved = False
        # Try perturbing each point
        for i in range(N):
            if time.time() > deadline - 10:
                break
            for _ in range(3):
                perturbation = np.random.normal(0, 0.05, 2)
                pts_new = best_pts.copy()
                pts_new[i] += perturbation
                s_new = _score(pts_new)
                if s_new > best_score:
                    best_score = s_new
                    best_pts = pts_new.copy()
                    improved = True
    
    return best_pts
# EVOLVE-BLOCK-END
