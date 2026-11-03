"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import random
import time
from scipy.spatial import ConvexHull

# Precompute all triangle indices
TRI_INDICES = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            TRI_INDICES.append((i, j, k))
TRI_INDICES = np.array(TRI_INDICES)


def _triangle_area(p1, p2, p3):
    """Compute area of triangle given three 2D points."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    return 0.5 * abs(ax*(by - cy) + bx*(cy - ay) + cx*(ay - by))


def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples."""
    areas = np.zeros(len(TRI_INDICES))
    for idx, (i, j, k) in enumerate(TRI_INDICES):
        ax, ay = pts[i]
        bx, by = pts[j]
        cx, cy = pts[k]
        areas[idx] = 0.5 * abs(ax*(by - cy) + bx*(cy - ay) + cx*(ay - by))
    return np.min(areas)


def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume = area
    except Exception:
        # Fallback: if points are collinear or degenerate
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-10:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area


def _fermat_spiral(n, scale=1.0):
    """Generate points on a Fermat/Vogel spiral."""
    i = np.arange(n)
    r = np.sqrt(i / n) * scale
    theta = i * 2.399963229728653  # Golden angle in radians
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.column_stack((x, y))
    points -= np.mean(points, axis=0)
    return points


def _regular_polygon(n, radius=1.0):
    """Generate points on a regular polygon with center point if odd."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    if n % 2 == 1:
        # Place (n-1)/2 pairs and one center
        n_outer = n - 1
        angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
        x = radius * np.cos(angles_outer)
        y = radius * np.sin(angles_outer)
        points = np.column_stack((x, y))
        points = np.vstack([points, [[0.0, 0.0]]])
    else:
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        points = np.column_stack((x, y))
    points -= np.mean(points, axis=0)
    return points


def _grid_init(n):
    """Generate points on a grid."""
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= n:
                break
            pts.append([(i - side/2 + 0.5) * 0.5, (j - side/2 + 0.5) * 0.5])
            count += 1
        if count >= n:
            break
    points = np.array(pts)
    points -= np.mean(points, axis=0)
    return points


def _random_init(n, seed):
    """Generate random points in a square."""
    rng = random.Random(seed)
    points = np.array([[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(n)])
    points -= np.mean(points, axis=0)
    return points


def _sa_optimize(pts, max_iter=5000, t_start=0.01, t_end=0.0001, rng_seed=42):
    """Simulated annealing with 2-point moves."""
    rng = random.Random(rng_seed)
    current = pts.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    n = len(pts)
    for it in range(max_iter):
        t = t_start * (t_end / t_start) ** (it / max_iter)
        
        # Random 2-point move: pick two points, perturb them
        i, j = rng.sample(range(n), 2)
        new_pts = current.copy()
        dx1 = rng.gauss(0, t)
        dy1 = rng.gauss(0, t)
        dx2 = rng.gauss(0, t)
        dy2 = rng.gauss(0, t)
        new_pts[i, 0] += dx1
        new_pts[i, 1] += dy1
        new_pts[j, 0] += dx2
        new_pts[j, 1] += dy2
        
        new_score = _score(new_pts)
        
        if new_score >= current_score or rng.random() < np.exp((new_score - current_score) / max(t, 1e-10)):
            current = new_pts
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
    
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Multiple diverse initializations
    inits = [
        _fermat_spiral(N, scale=1.0),
        _fermat_spiral(N, scale=0.8),
        _regular_polygon(N, radius=1.0),
        _grid_init(N),
        _random_init(N, seed=1),
        _random_init(N, seed=42),
        _random_init(N, seed=123),
    ]
    
    best_pts = None
    best_score = 0.0
    
    # Evaluate all initializations
    for pts in inits:
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Simulated annealing from each initialization (with time check)
    for idx, pts in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
        
        # Run SA from this initialization
        sa_pts, sa_score = _sa_optimize(pts, max_iter=3000, t_start=0.02, t_end=0.0005, rng_seed=idx*100+7)
        
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts.copy()
        
        # Check time
        if time.time() > deadline - 10.0:
            break
    
    # Final local refinement from best
    if time.time() < deadline - 5.0 and best_pts is not None:
        final_pts, final_score = _sa_optimize(best_pts, max_iter=2000, t_start=0.005, t_end=0.0001, rng_seed=999)
        if final_score > best_score:
            best_pts = final_pts
            best_score = final_score
    
    return best_pts
# EVOLVE-BLOCK-END
