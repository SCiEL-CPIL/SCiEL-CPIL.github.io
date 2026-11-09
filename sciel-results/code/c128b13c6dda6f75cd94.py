import numpy as np
import time

def _vertex_score(v, pts):
    """Compute the vertex score for vertex v given all points pts."""
    diffs = pts - v
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    # Exclude self
    mask = np.ones(len(pts), dtype=bool)
    mask[np.argmin(dists)] = False  # self distance is 0
    d = np.sort(dists[mask])
    # d has 19 elements
    if np.any(d < 0.05):
        return 0.0
    gaps = np.diff(d)
    k = np.argmin(gaps)
    D = (d[k] + d[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k-1:k+3]
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _min_vertex_score(pts):
    scores = [_vertex_score(pts[i], pts) for i in range(len(pts))]
    return min(scores)

def _is_valid(pts):
    if len(pts) != 20:
        return False
    if not np.all(np.isfinite(pts)):
        return False
    # Check no repeated points
    for i in range(20):
        for j in range(i+1, 20):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                return False
    # Check convexity: all cross products same sign
    signs = []
    for i in range(20):
        p0 = pts[i]
        p1 = pts[(i+1) % 20]
        p2 = pts[(i+2) % 20]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross) < 1e-12:
            return False
        signs.append(np.sign(cross))
    if not all(s == signs[0] for s in signs):
        return False
    return True

def _normalize(pts):
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    dists = np.linalg.norm(pts_norm, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return pts_norm, False
    pts_norm = pts_norm / mean_dist
    return pts_norm, True

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = 620.0  # safety margin
    
    # Start with regular 20-gon
    n = 20
    angles = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    norm_pts, valid = _normalize(pts)
    if valid and _is_valid(norm_pts):
        best_score = _min_vertex_score(norm_pts)
        best_pts = pts.copy()
    else:
        best_score = 0.0
        best_pts = pts.copy()
    
    # Try to improve by perturbing
    rng = np.random.default_rng(42)
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Small perturbation
        perturb = rng.normal(0, 0.01, size=(20, 2))
        trial = pts + perturb
        
        norm_trial, valid = _normalize(trial)
        if not valid or not _is_valid(norm_trial):
            continue
        
        score = _min_vertex_score(norm_trial)
        if score > best_score:
            best_score = score
            best_pts = trial.copy()
            pts = trial.copy()
    
    return best_pts
