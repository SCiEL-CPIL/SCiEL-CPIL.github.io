import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

N = 14
START_TIME = time.time()
DEADLINE = START_TIME + 115.0

def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

TRIPLE_INDICES = _get_triple_indices(N)
I_IDX = TRIPLE_INDICES[:, 0]
J_IDX = TRIPLE_INDICES[:, 1]
K_IDX = TRIPLE_INDICES[:, 2]
NUM_TRIPLES = len(TRIPLE_INDICES)

def _score(points):
    """Compute min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    p_i = points[I_IDX]
    p_j = points[J_IDX]
    p_k = points[K_IDX]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _neg_score(flat_points):
    points = flat_points.reshape((N, 2))
    return -_score(points)

def _generate_ring_config(rings):
    """
    Generate points from ring configuration.
    rings: list of (count, radius, angle_offset)
    """
    pts = []
    for count, radius, offset in rings:
        angles = np.linspace(offset, 2*np.pi + offset, count, endpoint=False)
        pts.append(np.column_stack([radius * np.cos(angles), radius * np.sin(angles)]))
    return np.vstack(pts)

def _make_regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def construct():
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0
    
    np.random.seed(42)
    
    candidates = []
    
    # 1. Regular 14-gon
    candidates.append(_make_regular_polygon(N))
    
    # 2. Concentric Rings
    # Try various splits and radius ratios
    splits = [
        (7, 7),
        (6, 8),
        (8, 6),
        (5, 9),
        (9, 5),
        (4, 10),
        (10, 4),
        (5, 5, 4),
        (4, 5, 5),
        (5, 4, 5),
        (3, 5, 6),
        (6, 5, 3),
        (7, 4, 3),
        (3, 4, 7),
        (4, 7, 3),
        (7, 3, 4),
        (2, 6, 6),
        (6, 2, 6),
        (6, 6, 2),
        (1, 6, 7),
        (7, 1, 6),
        (6, 7, 1),
    ]
    
    radius_ratios = [0.3, 0.5, 0.7, 0.8, 0.9]
    
    for split in splits:
        if len(split) == 2:
            n1, n2 = split
            for r1 in radius_ratios:
                # Outer ring usually has radius 1.0
                # Inner ring radius r1
                # Try different angular offsets
                for offset in [0, np.pi/n1, np.pi/n2]:
                    rings = [(n1, 1.0, 0), (n2, r1, offset)]
                    candidates.append(_generate_ring_config(rings))
        elif len(split) == 3:
            n1, n2, n3 = split
            for r1 in [0.3, 0.5, 0.7]:
                for r2 in [0.6, 0.8]:
                    if r1 < r2:
                        rings = [(n1, r1, 0), (n2, r2, np.pi/n2), (n3, 1.0, np.pi/n3)]
                        candidates.append(_generate_ring_config(rings))
                        # Swap offsets
                        rings = [(n1, r1, np.pi/n1), (n2, r2, 0), (n3, 1.0, np.pi/n3)]
                        candidates.append(_generate_ring_config(rings))
    
    # 3. Random Points
    for _ in range(20):
        candidates.append(np.random.randn(N, 2) * 0.8)
        
    # 4. Perturbed Regular Polygon
    base = _make_regular_polygon(N)
    for _ in range(10):
        perturbed = base + np.random.randn(N, 2) * 0.2
        candidates.append(perturbed)

    # Evaluate all candidates
    scored_candidates = []
    for c in candidates:
        s = _score(c)
        scored_candidates.append((s, c))
        
    scored_candidates.sort(key=lambda x: -x[0])
    
    best_score = -1
    best_points = None
    
    # Optimize top candidates
    # We have ~115 seconds. 
    # _score is fast. minimize is slower.
    # Let's take top 10 candidates and optimize them.
    
    top_k = 10
    for i in range(top_k):
        if time.time() > DEADLINE - 10:
            break
            
        s, c = scored_candidates[i]
        if s > best_score:
            best_score = s
            best_points = c.copy()
            
        # Optimize this candidate
        try:
            # Nelder-Mead is robust
            res = minimize(
                _neg_score,
                c.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'maxfev': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            opt_points = res.x.reshape((N, 2))
            opt_score = _score(opt_points)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
                
        except Exception:
            pass
            
    # Final polish on the best point
    if best_points is not None and time.time() < DEADLINE - 5:
        try:
            # L-BFGS-B with numerical gradient for final precision
            res = minimize(
                _neg_score,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-12}
            )
            final_points = res.x.reshape((N, 2))
            final_score = _score(final_points)
            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = _make_regular_polygon(N)
        
    return best_points
