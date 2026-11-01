import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

# Pre-compute triangle indices for speed
def _get_triangle_indices(n):
    """Pre-compute all (i,j,k) with i<j<k"""
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(points):
    """Compute minimum triangle area for given points (assumes valid shape)."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _neg_score(points_flat):
    """Negative score for minimization."""
    points = points_flat.reshape(-1, 2)
    return -_score(points)

def _normalize(points):
    """Scale so convex hull area = 1."""
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    """Regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _concentric_circles(n, seed=42):
    """Points on two concentric circles."""
    rng = np.random.RandomState(seed)
    # 7 points on outer, 7 on inner
    outer_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + rng.uniform(0, 2 * np.pi)
    inner_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + rng.uniform(0, 2 * np.pi)
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner = np.column_stack([0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)])
    pts = np.vstack([outer, inner])
    return pts

def _random_in_disk(n, seed=None):
    """Random points in unit disk."""
    rng = np.random.RandomState(seed)
    r = np.sqrt(rng.uniform(0, 1, n))
    theta = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

def _perturbed_polygon(n, base_radius=1.0, noise_scale=0.05, seed=None, rotation=0.0):
    """Perturbed regular polygon."""
    rng = np.random.RandomState(seed)
    base = _regular_polygon(n, base_radius, rotation)
    noise = rng.uniform(-noise_scale, noise_scale, (n, 2))
    return base + noise

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_points = None
    
    def _try_optimize(init_points, method='Nelder-Mead', maxiter=2000, ftol=1e-10, xtol=1e-8):
        """Try optimizing from init_points, return (score, points) or None."""
        nonlocal best_score, best_points
        if time.time() > deadline:
            return None
        try:
            init_norm = _normalize(init_points)
            init_score = _score(init_norm)
            
            if method == 'Nelder-Mead':
                res = scipy.optimize.minimize(
                    _neg_score,
                    init_norm.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': xtol, 'fatol': ftol, 'adaptive': True}
                )
            elif method == 'Powell':
                res = scipy.optimize.minimize(
                    _neg_score,
                    init_norm.flatten(),
                    method='Powell',
                    options={'maxiter': maxiter, 'ftol': ftol, 'xtol': xtol}
                )
            else:
                return None
            
            opt_pts = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt_pts)
            
            if sc > best_score:
                best_score = sc
                best_points = opt_pts.copy()
            
            return (sc, opt_pts)
        except Exception:
            return None
    
    # Generate diverse initializations
    inits = []
    
    # 1. Regular polygon with various rotations
    for rot in [0.0, np.pi / 14, np.pi / 7, np.pi / 3]:
        inits.append(_regular_polygon(N, 1.0, rot))
    
    # 2. Perturbed regular polygons with different noise levels and seeds
    for seed in range(10):
        for ns in [0.03, 0.08, 0.15]:
            inits.append(_perturbed_polygon(N, 1.0, ns, seed=seed + 100, rotation=seed * 0.1))
    
    # 3. Concentric circles
    for seed in range(5):
        inits.append(_concentric_circles(N, seed=seed))
    
    # 4. Random in disk
    for seed in range(5):
        inits.append(_random_in_disk(N, seed=seed))
    
    # 5. Points on two circles with different radii ratios
    for ratio in [0.3, 0.4, 0.6, 0.7]:
        n_outer = 8
        n_inner = 6
        outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
        inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
        outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        inner = np.column_stack([ratio * np.cos(inner_angles), ratio * np.sin(inner_angles)])
        inits.append(np.vstack([outer, inner]))
    
    # Sort inits by initial score (best first) to prioritize good starts
    scored_inits = []
    for init in inits:
        try:
            init_norm = _normalize(init)
            sc = _score(init_norm)
            scored_inits.append((sc, init_norm))
        except Exception:
            continue
    
    scored_inits.sort(key=lambda x: -x[0])
    
    # Update best with best initial
    if scored_inits:
        best_init_score, best_init_pts = scored_inits[0]
        if best_init_score > best_score:
            best_score = best_init_score
            best_points = best_init_pts.copy()
    
    # Optimize from top initializations
    for rank, (init_score, init_pts) in enumerate(scored_inits):
        if time.time() > deadline:
            break
        
        # Try Nelder-Mead first (faster per iteration, good for low-dim)
        time_left = deadline - time.time()
        if time_left > 5:
            _try_optimize(init_pts, 'Nelder-Mead', maxiter=3000, ftol=1e-11, xtol=1e-9)
        
        # Try Powell if time permits
        time_left = deadline - time.time()
        if time_left > 8 and rank < 15:
            _try_optimize(init_pts, 'Powell', maxiter=500, ftol=1e-10, xtol=1e-8)
    
    # If we still have time, do a few more random restarts with optimization
    while time.time() < deadline - 10:
        seed = int(time.time() * 1000) % 100000
        init = _perturbed_polygon(N, 1.0, 0.1, seed=seed, rotation=seed * 0.01)
        _try_optimize(init, 'Nelder-Mead', maxiter=2000, ftol=1e-11, xtol=1e-9)
    
    # Final fallback
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0, 0.0))
    else:
        best_points = _normalize(best_points)
    
    return best_points
