import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 620.0

def _vertex_score(distances):
    """distances: 19 values (distances from v to other vertices), unsorted."""
    d = np.sort(distances)
    if np.any(d < 0.05):
        return 0.0
    gaps = d[1:] - d[:-1]
    # first index minimizing gap
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        sel = d[0:4]
    elif k == 17:
        sel = d[15:19]
    else:
        sel = d[k - 1:k + 3]
    ratio = np.maximum(sel / D, D / sel)
    raw = np.mean(ratio)
    return min(1.0, 1.0 / raw)

def _min_vertex_score(polygon):
    n = len(polygon)
    scores = []
    for i in range(n):
        dists = np.linalg.norm(polygon - polygon[i], axis=1)
        dists = np.delete(dists, i)
        scores.append(_vertex_score(dists))
    return float(np.min(scores))

def _is_valid(polygon, n=20):
    """Contract validity: finite, strictly convex (no repeats, all turns same
    nonzero sign, total turning +-2*pi)."""
    if not np.all(np.isfinite(polygon)):
        return False
    # repeated points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-12:
                return False
    # edges / turns
    pts = np.vstack([polygon, polygon[:1]])
    edges = pts[1:] - pts[:-1]
    # no zero-length edges
    if np.any(np.linalg.norm(edges, axis=1) < 1e-12):
        return False
    cross = np.cross(edges[:-1], edges[1:])
    if np.any(np.abs(cross) < 1e-12):
        return False
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return False
    # total turning +-2*pi
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    dangles = np.diff(angles)
    dangles = np.append(dangles, angles[0] - angles[-1])
    dangles = np.arctan2(np.sin(dangles), np.cos(dangles))
    total = np.sum(dangles)
    if abs(abs(total) - 2 * np.pi) > 1e-6:
        return False
    return True

def _normalized(polygon):
    c = np.mean(polygon, axis=0)
    p = polygon - c
    m = np.mean(np.linalg.norm(p, axis=1))
    if m <= 0.01:
        return None
    return p / m

def _objective(params):
    """
    Objective function for optimizer.
    Returns negative min vertex score, plus penalty for invalidity.
    """
    n = 20
    polygon = params.reshape(n, 2)
    
    # Check validity
    if not _is_valid(polygon):
        return 1000.0  # Large penalty
    
    nm = _normalized(polygon)
    if nm is None:
        return 1000.0
        
    score = _min_vertex_score(nm)
    return -score

def _generate_random_convex(rng):
    """Generate a random convex polygon."""
    n = 20
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    radii = rng.uniform(0.5, 1.5, n)
    pts = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    return pts

def equidistant_points_polygon20() -> np.ndarray:
    best = None
    best_score = -1.0
    
    # Seed with regular polygon
    a = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    reg = np.stack([np.cos(a), np.sin(a)], axis=1)
    best = reg.copy()
    best_score = _min_vertex_score(_normalized(reg))
    
    # Local optimization from regular polygon
    if time.time() < _DEADLINE - 60:
        try:
            res = minimize(_objective, reg.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < -best_score: # i.e. -score < -best_score => score > best_score
                trial = res.x.reshape(20, 2)
                if _is_valid(trial):
                    nm = _normalized(trial)
                    if nm is not None:
                        s = _min_vertex_score(nm)
                        if s > best_score:
                            best_score = s
                            best = nm
        except Exception:
            pass
            
    # Try several random starts
    rng = np.random.default_rng(42)
    for i in range(10):
        if time.time() > _DEADLINE - 30:
            break
        start = _generate_random_convex(rng)
        if not _is_valid(start):
            continue
        try:
            res = minimize(_objective, start.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < -best_score:
                trial = res.x.reshape(20, 2)
                if _is_valid(trial):
                    nm = _normalized(trial)
                    if nm is not None:
                        s = _min_vertex_score(nm)
                        if s > best_score:
                            best_score = s
                            best = nm
        except Exception:
            continue

    return best
