"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations


def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples of points."""
    n = len(pts)
    # Vectorized: compute all cross products
    # For each pair (i,j), compute cross with all k
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            # Cross product with all other points
            diffs = pts - pts[i]
            crosses = np.abs(diffs[:, 0] * v1[1] - diffs[:, 1] * v1[0])
            # Exclude i and j themselves
            crosses[i] = float('inf')
            crosses[j] = float('inf')
            local_min = np.min(crosses)
            if local_min < min_area:
                min_area = local_min
    return 0.5 * min_area


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume = area
    except Exception:
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    min_ta = _min_triangle_area(pts)
    return min_ta / hull_a


def _neg_score(pts):
    return -_score(pts)


def _symmetric_ring_init(center, radius, n_inner, n_outer):
    """
    Create a symmetric ring initialization:
    - 1 center point
    - n_inner points on inner ring
    - n_outer points on outer ring
    Total: 1 + n_inner + n_outer = 13
    """
    pts = [np.array(center, dtype=float)]
    
    # Inner ring
    if n_inner > 0:
        angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
        for a in angles_inner:
            pts.append(center + radius * 0.5 * np.array([np.cos(a), np.sin(a)]))
    
    # Outer ring
    if n_outer > 0:
        angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
        # Slight offset to avoid perfect alignment with inner ring
        offset = np.pi / n_outer if n_outer > 1 else 0
        for a in angles_outer:
            pts.append(center + radius * np.array([np.cos(a + offset), np.sin(a + offset)]))
    
    return np.array(pts)


def _optimize_from_init(pts0, deadline):
    """Run L-BFGS-B optimization from initial point."""
    x0 = pts0.flatten()
    
    def obj(x):
        pts = x.reshape(-1, 2)
        return _neg_score(pts)
    
    result = minimize(obj, x0, method='L-BFGS-B', 
                      options={'maxiter': 200, 'gtol': 1e-8})
    return result.x.reshape(-1, 2), _score(result.x.reshape(-1, 2))


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_pts = None
    best_score = -1.0
    
    # Generate diverse symmetric ring initializations
    inits = []
    
    # Configuration 1: 1 center + 12 outer
    for center in [np.array([0.5, 0.5]), np.array([0.4, 0.5]), np.array([0.5, 0.4])]:
        for radius in [0.4, 0.45, 0.5]:
            pts = _symmetric_ring_init(center, radius, 0, 12)
            inits.append(pts)
    
    # Configuration 2: 1 center + 6 inner + 6 outer
    for center in [np.array([0.5, 0.5]), np.array([0.45, 0.5])]:
        for radius in [0.4, 0.45, 0.5]:
            pts = _symmetric_ring_init(center, radius, 6, 6)
            inits.append(pts)
    
    # Configuration 3: 1 center + 4 inner + 8 outer
    for center in [np.array([0.5, 0.5])]:
        for radius in [0.45, 0.5]:
            pts = _symmetric_ring_init(center, radius, 4, 8)
            inits.append(pts)
    
    # Also add some perturbed versions for diversity
    rng = np.random.RandomState(42)
    for base in inits[:6]:
        perturbed = base + rng.randn(13, 2) * 0.02
        inits.append(perturbed)
    
    # Try each initialization
    for init_pts in inits:
        if time.time() > deadline:
            break
        
        try:
            optimized_pts, score = _optimize_from_init(init_pts, deadline)
            if score > best_score:
                best_score = score
                best_pts = optimized_pts
        except Exception:
            continue
    
    # If no good result, fall back to best initialization
    if best_pts is None:
        best_pts = inits[0]
        best_score = _score(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
