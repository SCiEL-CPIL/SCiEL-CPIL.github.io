"""Heilbronn convex regions: maximize min triangle area / hull area for 13 points."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13

def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # Compute all triangle areas using cross product
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                if cross < min_area:
                    min_area = cross
    
    if min_area == float('inf'):
        return 0.0
    
    return (0.5 * min_area) / hull_area


def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(-1, 2)
    return -_score_points(pts)


def _make_initial_candidates(rng):
    """Generate diverse initial configurations."""
    candidates = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(reg.copy())
    
    # 2. Jittered regular polygon
    for _ in range(5):
        jitter = rng.random() * 0.1
        jitters = rng.uniform(-jitter, jitter, size=(N, 2))
        candidates.append(reg + jitters)
    
    # 3. Fibonacci spiral
    golden_angle = np.pi * (3 - np.sqrt(5))
    fib = np.zeros((N, 2))
    for i in range(N):
        r = np.sqrt((i + 0.5) / N)
        theta = i * golden_angle
        fib[i] = [r * np.cos(theta), r * np.sin(theta)]
    candidates.append(fib)
    
    # 4. Jittered Fibonacci
    for _ in range(5):
        jitter = rng.random() * 0.05
        jitters = rng.uniform(-jitter, jitter, size=(N, 2))
        candidates.append(fib + jitters)
    
    # 5. Concentric rings
    for _ in range(3):
        inner = 3
        outer = N - inner
        inner_angles = np.linspace(0, 2 * np.pi, inner, endpoint=False)
        outer_angles = np.linspace(0, 2 * np.pi, outer, endpoint=False)
        pts = np.zeros((N, 2))
        pts[:inner] = np.column_stack([0.3 * np.cos(inner_angles), 0.3 * np.sin(inner_angles)])
        pts[inner:] = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        rng.shuffle(pts)
        candidates.append(pts)
    
    # 6. Random uniform in unit disk
    for _ in range(10):
        r = np.sqrt(rng.random(N))
        theta = 2 * np.pi * rng.random(N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        candidates.append(pts)
    
    # 7. Grid-like
    for _ in range(3):
        # 4x4 grid with one removed, scaled
        gx = np.array([0, 0.33, 0.66, 1.0])
        gy = np.array([0, 0.33, 0.66, 1.0])
        grid_pts = np.array([[x, y] for x in gx for y in gy])
        # Remove one point to get 13
        idx = rng.randint(0, len(grid_pts) - 1)
        grid_pts = np.delete(grid_pts, idx, axis=0)
        # Center and scale
        grid_pts = (grid_pts - grid_pts.mean(axis=0)) / (np.max(grid_pts) - np.min(grid_pts))
        grid_pts = 0.5 * grid_pts + 0.5
        jitter = rng.uniform(-0.02, 0.02, size=(N, 2))
        candidates.append(grid_pts + jitter)
    
    return candidates


def construct():
    """Main entry point: find 13 points maximizing min triangle area / hull area."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.RandomState(42)
    
    best_score = 0.0
    best_pts = None
    
    # Generate initial candidates
    candidates = _make_initial_candidates(rng)
    
    # Also add some random restarts
    for _ in range(5):
        r = np.sqrt(rng.random(N))
        theta = 2 * np.pi * rng.random(N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        candidates.append(pts)
    
    # Evaluate all candidates
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 5.0:
            break
        score = _score_points(cand)
        if score > best_score:
            best_score = score
            best_pts = cand.copy()
    
    # Local refinement with Nelder-Mead on top candidates
    # Sort candidates by score
    scored = []
    for cand in candidates:
        s = _score_points(cand)
        scored.append((s, cand))
    scored.sort(key=lambda x: -x[0])
    
    # Refine top candidates
    top_k = min(10, len(scored))
    for s, cand in scored[:top_k]:
        if time.time() > deadline - 10.0:
            break
        
        pts_flat = cand.flatten()
        try:
            result = minimize(
                _neg_score,
                pts_flat,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'maxfev': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            refined_pts = result.x.reshape(-1, 2)
            refined_score = _score_points(refined_pts)
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined_pts.copy()
        except Exception:
            pass
    
    # If we still have time, do a few more random restarts with refinement
    while time.time() < deadline - 15.0:
        r = np.sqrt(rng.random(N))
        theta = 2 * np.pi * rng.random(N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        score = _score_points(pts)
        if score > best_score * 0.9:  # Only refine if competitive
            pts_flat = pts.flatten()
            try:
                result = minimize(
                    _neg_score,
                    pts_flat,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 300,
                        'maxfev': 300,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                refined_pts = result.x.reshape(-1, 2)
                refined_score = _score_points(refined_pts)
                if refined_score > best_score:
                    best_score = refined_score
                    best_pts = refined_pts.copy()
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts
