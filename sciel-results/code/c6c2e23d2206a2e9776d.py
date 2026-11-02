import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13
DEADLINE = 115.0  # Safety margin before 120s kill

# Pre-compute all triples for fast vectorized scoring
# There are C(13, 3) = 286 triples
_triple_i = []
_triple_j = []
_triple_k = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _triple_i.append(_i)
            _triple_j.append(_j)
            _triple_k.append(_k)

_triple_i = np.array(_triple_i, dtype=np.int32)
_triple_j = np.array(_triple_j, dtype=np.int32)
_triple_k = np.array(_triple_k, dtype=np.int32)

def _compute_areas(pts):
    """Compute all triangle areas for a point set."""
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-12

def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    
    areas = _compute_areas(pts)
    min_area = np.min(areas)
    
    if min_area < 1e-15:
        return 0.0
    
    return min_area / hull_area

def _soft_min_score(pts, beta=100.0):
    """
    Compute a smooth approximation of the min triangle area / hull area.
    """
    pts = np.asarray(pts, dtype=np.float64)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return -np.inf
        
    areas = _compute_areas(pts)
    
    vals = -beta * areas
    v_max = np.max(vals)
    if v_max == -np.inf:
        return -np.inf
        
    log_sum = v_max + np.log(np.sum(np.exp(vals - v_max)))
    soft_min_val = -log_sum / beta
    
    return soft_min_val / hull_area

def _neg_soft_score(params, beta):
    """Negative soft score for minimization."""
    pts = params.reshape(N, 2)
    return -_soft_min_score(pts, beta)

def _build_rings(radii, angles, sizes):
    """
    Build point set from concentric regular polygons.
    radii: array of radii for each ring
    angles: array of rotation angles for each ring
    sizes: list of number of points in each ring
    """
    pts = []
    for i, (r, a, size) in enumerate(zip(radii, angles, sizes)):
        if size == 0:
            continue
        if size == 1:
            # Single point, usually at center if r=0, but can be anywhere
            # If r>0 and size==1, it's just a point at (r*cos(a), r*sin(a))
            pts.append([r * np.cos(a), r * np.sin(a)])
        else:
            theta = np.linspace(0, 2 * np.pi, size, endpoint=False) + a
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            pts.append(np.column_stack([x, y]))
    
    return np.vstack(pts)

def _neg_ring_score(params, sizes, beta=50.0):
    """
    Objective function for ring-based optimization.
    params: [r1, a1, r2, a2, ...]
    sizes: list of points per ring
    """
    K = len(sizes)
    radii = params[0::2]
    angles = params[1::2]
    
    # Ensure radii are non-negative
    radii = np.abs(radii)
    
    pts = _build_rings(radii, angles, sizes)
    
    if len(pts) != N:
        return 1e10
        
    score = _soft_min_score(pts, beta)
    if np.isinf(score):
        return 1e10
    return -score

def _optimize_ring_structure(sizes, t_start):
    """
    Optimize a specific ring structure.
    Returns best points and score.
    """
    K = len(sizes)
    # Initial guess: equal radii spacing, random angles
    rng = np.random.RandomState(42)
    
    # We try multiple initializations
    best_pts = None
    best_score = -np.inf
    
    # Number of restarts depends on time available
    time_left = DEADLINE - (time.time() - t_start)
    max_restarts = max(5, int(time_left / 2.0)) # ~2s per restart
    
    for restart in range(max_restarts):
        if (time.time() - t_start) > (DEADLINE - 10.0):
            break
            
        # Initialize parameters
        # Radii should be increasing
        base_radii = np.linspace(0.1, 1.0, K)
        # Add some noise
        radii_init = base_radii * (1.0 + 0.2 * rng.randn(K))
        angles_init = rng.uniform(0, 2 * np.pi, K)
        
        # Flatten params
        params_init = np.zeros(2 * K)
        params_init[0::2] = radii_init
        params_init[1::2] = angles_init
        
        # Bounds
        # Radii: [0, 2] (allow some flexibility)
        # Angles: [0, 2*pi]
        bounds = []
        for _ in range(K):
            bounds.append((0.0, 2.0))  # Radius
            bounds.append((0.0, 2 * np.pi)) # Angle
            
        # Optimize
        try:
            # Use L-BFGS-B with finite differences
            res = minimize(
                lambda x: _neg_ring_score(x, sizes, beta=100.0),
                params_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            # Evaluate hard score
            params_opt = res.x
            radii_opt = np.abs(params_opt[0::2])
            angles_opt = params_opt[1::2]
            pts_opt = _build_rings(radii_opt, angles_opt, sizes)
            
            score = _score_points(pts_opt)
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt
                
        except Exception as e:
            pass
            
    return best_pts, best_score

def _full_polish(pts_init, t_start):
    """
    Perform a final optimization on all 26 coordinates starting from pts_init.
    """
    if (time.time() - t_start) > (DEADLINE - 5.0):
        return pts_init, _score_points(pts_init)
        
    params_init = pts_init.flatten()
    
    # Use Nelder-Mead for robustness, or L-BFGS-B with bounds
    # L-BFGS-B is faster if we have good gradient estimates, but finite diff is expensive.
    # Let's try Nelder-Mead first with a tight tolerance.
    
    try:
        res = minimize(
            lambda x: -_soft_min_score(x.reshape(N, 2), beta=200.0),
            params_init,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        pts_opt = res.x.reshape(N, 2)
        score = _score_points(pts_opt)
        return pts_opt, score
    except Exception:
        return pts_init, _score_points(pts_init)

def construct():
    t_start = time.time()
    
    best_pts = None
    best_score = -np.inf
    
    # Define possible ring structures for N=13
    # (sizes, description)
    structures = [
        ([1, 12], "1 center + 12 outer"),
        ([1, 6, 6], "1 center + 6 mid + 6 outer"),
        ([1, 4, 8], "1 center + 4 inner + 8 outer"),
        ([1, 5, 7], "1 center + 5 inner + 7 outer"),
        ([1, 3, 9], "1 center + 3 inner + 9 outer"),
        ([2, 11], "2 centers + 11 outer"),
        ([3, 10], "3 centers + 10 outer"),
        ([4, 9], "4 centers + 9 outer"),
        ([5, 8], "5 centers + 8 outer"),
        ([6, 7], "6 centers + 7 outer"),
        ([1, 2, 3, 7], "1+2+3+7"),
        ([1, 2, 4, 6], "1+2+4+6"),
        ([1, 3, 4, 5], "1+3+4+5"),
        ([1, 1, 1, 10], "3 centers + 10 outer"),
    ]
    
    # Allocate time for each structure
    time_per_structure = (DEADLINE - 20.0) / len(structures) # 20s reserved for polish
    
    for sizes, desc in structures:
        if (time.time() - t_start) > (DEADLINE - 10.0):
            break
            
        # Optimize this structure
        pts, score = _optimize_ring_structure(sizes, t_start)
        
        if pts is not None and score > best_score:
            best_score = score
            best_pts = pts
            
        # Print progress if needed (commented out for speed)
        # print(f"Structure {desc}: {score:.6f}")
        
    # Final Polish
    if best_pts is not None:
        # Try polishing the best found
        polished_pts, polished_score = _full_polish(best_pts, t_start)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts
            
        # Also try polishing a few other good candidates if time permits
        # Re-run optimization for top 2 structures with more restarts? 
        # Or just polish the initial regular polygon?
        
        # Let's also try a standard regular 13-gon polish
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        p_reg = np.column_stack([np.cos(angles), np.sin(angles)])
        polished_reg, score_reg = _full_polish(p_reg, t_start)
        if score_reg > best_score:
            best_score = score_reg
            best_pts = polished_reg
            
    if best_pts is None:
        # Fallback
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_pts
