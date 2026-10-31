import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    """Compute all triangle areas using precomputed indices."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _objective(x_flat):
    """
    Direct objective: minimize negative score.
    No rotation normalization - the score is invariant under rigid transforms.
    """
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _smooth_objective(x_flat, p=30):
    """
    Smooth surrogate using power mean of inverse triangle areas.
    """
    points = x_flat.reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    
    inv_areas = 1.0 / areas
    soft_min = (np.mean(inv_areas ** p)) ** (-1.0 / p)
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def _coordinate_descent_polish(points, max_iters=300, deadline=None):
    """
    Coordinate descent: optimize each point's x and y with adaptive step size.
    """
    pts = points.copy()
    best_s = _score(pts)
    step = 0.05
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 2:
            break
        
        improved = False
        for i in range(N):
            # Try x adjustments
            for dx in [step, -step, step*0.5, -step*0.5, step*0.1, -step*0.1]:
                trial = pts.copy()
                trial[i, 0] += dx
                s = _score(trial)
                if s > best_s + 1e-12:
                    pts = trial
                    best_s = s
                    improved = True
                    break
            
            # Try y adjustments
            for dy in [step, -step, step*0.5, -step*0.5, step*0.1, -step*0.1]:
                trial = pts.copy()
                trial[i, 1] += dy
                s = _score(trial)
                if s > best_s + 1e-12:
                    pts = trial
                    best_s = s
                    improved = True
                    break
        
        if not improved:
            step *= 0.5
            if step < 1e-12:
                break
    
    return pts, best_s

def _local_polish(points, deadline):
    """
    Multi-method local polish: Nelder-Mead followed by coordinate descent.
    """
    best_pts = points.copy()
    best_s = _score(best_pts)
    
    # Nelder-Mead polish
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _objective,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_pts = res.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    # Coordinate descent polish
    if time.time() < deadline - 3:
        cd_pts, cd_s = _coordinate_descent_polish(best_pts, max_iters=200, deadline=deadline)
        if cd_s > best_s:
            best_s = cd_s
            best_pts = cd_pts
    
    return best_pts, best_s

def _get_initial_candidates():
    """Generate diverse initial configurations."""
    candidates = []
    rng = np.random.default_rng(42)
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(pts)
    
    # 2. 4x4 grid minus one corner
    xs = np.array([-1.5, -0.5, 0.5, 1.5])
    ys = np.array([-1.5, -0.5, 0.5, 1.5])
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    pts = pts[:-1]
    pts = pts - np.mean(pts, axis=0)
    candidates.append(pts)
    
    # 3. 3x5 grid
    xs = np.array([-1.0, 0.0, 1.0])
    ys = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    pts = pts - np.mean(pts, axis=0)
    candidates.append(pts)
    
    # 4. 5x3 grid
    xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    ys = np.array([-1.0, 0.0, 1.0])
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    pts = pts - np.mean(pts, axis=0)
    candidates.append(pts)
    
    # 5. Hexagonal lattice subset
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + j/2.0
            y = j * np.sqrt(3)/2.0
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    dists = np.linalg.norm(hex_pts, axis=1)
    order = np.argsort(dists)
    pts = hex_pts[order[:13]]
    pts = pts - np.mean(pts, axis=0)
    candidates.append(pts)
    
    # 6. Two rings (7 outer, 6 inner)
    r_out, r_in = 1.0, 0.5
    ang_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    ang_in = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    pts = np.vstack([
        np.column_stack([r_out * np.cos(ang_out), r_out * np.sin(ang_out)]),
        np.column_stack([r_in * np.cos(ang_in), r_in * np.sin(ang_in)])
    ])
    candidates.append(pts)
    
    # 7-12. Random configurations
    for _ in range(6):
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        candidates.append(pts)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # Generate initial candidates
    candidates = _get_initial_candidates()
    
    # Phase 1: For each candidate, do smooth optimization then hard polish
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 25:
            break
        
        x0 = cand.flatten()
        
        # Quick smooth optimization to find a good basin
        try:
            res_smooth = minimize(
                lambda x: _smooth_objective(x, p=30),
                x0,
                method='Powell',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            _update_best(res_smooth.x.reshape(N, 2))
        except Exception:
            pass
        
        # Hard polish on the best found so far
        if time.time() < deadline - 5:
            start_x = res_smooth.x if 'res_smooth' in locals() else x0
            polished_pts, polished_s = _local_polish(start_x.reshape(N, 2), deadline)
            _update_best(polished_pts)
    
    # Phase 2: Differential Evolution for global search
    if time.time() < deadline - 30:
        try:
            bounds = [(-2.0, 2.0)] * (2 * N)
            
            de_res = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=60,
                popsize=20,
                mutation=(0.3, 0.9),
                recombination=0.8,
                tol=1e-8,
                polish=False,
                init='sobol'
            )
            
            if de_res.fun < 1e9:
                de_pts = de_res.x.reshape(N, 2)
                _update_best(de_pts)
                
                # Polish the DE result
                if time.time() < deadline - 5:
                    polished_pts, polished_s = _local_polish(de_pts, deadline)
                    _update_best(polished_pts)
                    
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish on the best candidate
    if best_points is not None and time.time() < deadline - 3:
        try:
            final_pts, final_s = _local_polish(best_points, deadline)
            _update_best(final_pts)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
