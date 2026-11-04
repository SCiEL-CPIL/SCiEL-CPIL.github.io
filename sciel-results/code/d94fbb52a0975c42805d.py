import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 14

# Precompute triple indices
_TRIPLE_INDICES = np.array(list(combinations(range(N), 3)), dtype=int)
_N_TRIPLES = _TRIPLE_INDICES.shape[0]

# Precompute which triples each point participates in for fast updates
_POINT_TRIPLES = [np.where((_TRIPLE_INDICES[:, 0] == i) | (_TRIPLE_INDICES[:, 1] == i) | (_TRIPLE_INDICES[:, 2] == i))[0] for i in range(N)]

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _triangle_areas(points):
    """Compute all triangle areas (unnormalized)."""
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
        
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _soft_min(areas, beta=20.0):
    """Smooth approximation of min using log-sum-exp."""
    m = np.min(areas)
    if m < 1e-15:
        return m
    a = areas - m
    log_sum = np.log(np.sum(np.exp(-beta * a)))
    return m - (1.0 / beta) * log_sum

def _neg_soft_score(points, beta=20.0):
    """Negative soft-min score for minimization."""
    areas = _triangle_areas(points)
    sm = _soft_min(areas, beta)
    ha = _hull_area(points)
    if ha < 1e-12:
        return 1e10
    return -sm / ha

def _neg_soft_score_grad(points, beta=20.0):
    """Gradient of negative soft-min score using finite differences."""
    areas = _triangle_areas(points)
    sm = _soft_min(areas, beta)
    ha = _hull_area(points)
    if ha < 1e-12:
        return np.zeros(2 * N)
    score = -sm / ha
    
    # Compute gradient via finite differences
    grad = np.zeros(2 * N)
    eps = 1e-6
    for i in range(N):
        for coord in range(2):
            idx = 2 * i + coord
            pts_plus = points.copy()
            pts_plus[i, coord] += eps
            s_plus = _neg_soft_score(pts_plus, beta)
            pts_minus = points.copy()
            pts_minus[i, coord] -= eps
            s_minus = _neg_soft_score(pts_minus, beta)
            grad[idx] = (s_plus - s_minus) / (2 * eps)
    
    return grad

def _fast_local_update(points, idx, delta):
    """
    Fast score update: only recompute triples affected by moving point idx.
    Returns the new score without full recomputation.
    """
    trial = points.copy()
    trial[idx] += delta
    
    # Recompute only affected triples
    affected = _POINT_TRIPLES[idx]
    i_idx = _TRIPLE_INDICES[affected, 0]
    j_idx = _TRIPLE_INDICES[affected, 1]
    k_idx = _TRIPLE_INDICES[affected, 2]
    
    p_i = trial[i_idx]
    p_j = trial[j_idx]
    p_k = trial[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    new_areas = 0.5 * np.abs(cross)
    
    # Get areas for non-affected triples from the original points
    mask = np.ones(_N_TRIPLES, dtype=bool)
    mask[affected] = False
    old_areas = _triangle_areas(points)[mask]
    
    all_areas = np.concatenate([old_areas, new_areas])
    min_area = np.min(all_areas)
    
    hull_area = _hull_area(trial)
    if hull_area < 1e-12:
        return 0.0, trial
    
    return min_area / hull_area, trial

def _local_search_fast(points, max_iters=500, step_size=0.01):
    """
    Fast coordinate-level local search using local-update scoring.
    """
    best_pts = points.copy()
    best_s = _score(best_pts)
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            for coord in range(2):
                for delta_val in [step_size, -step_size, step_size * 0.5, -step_size * 0.5]:
                    trial_pts = best_pts.copy()
                    trial_pts[i, coord] += delta_val
                    s, _ = _fast_local_update(best_pts, i, np.array([delta_val if coord == 0 else 0.0, delta_val if coord == 1 else 0.0]))
                    if s > best_s + 1e-10:
                        best_pts = trial_pts
                        best_s = s
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            step_size *= 0.5
            if step_size < 1e-5:
                break
    
    return best_pts, best_s

def _lbfgs_polish(points, deadline, max_iter=200):
    """
    L-BFGS-B polish using soft-min surrogate for gradient-based optimization.
    """
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(N, 2)
        return _neg_soft_score(pts, beta=30.0)
    
    def grad(x):
        pts = x.reshape(N, 2)
        return _neg_soft_score_grad(pts, beta=30.0)
    
    try:
        result = minimize(
            objective, x0, method='L-BFGS-B', jac=grad,
            options={'maxiter': max_iter, 'maxfun': 500}
        )
        if time.time() < deadline:
            polished = result.x.reshape(N, 2)
            s = _score(polished)
            if s > _score(points):
                return polished, s
    except Exception:
        pass
    
    return points, _score(points)

def _generate_ring_config(n_out, n_in, r, theta_offset):
    """Generate a two-ring configuration."""
    outer_angles = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    inner_angles = theta_offset + np.linspace(0, 2*np.pi, n_in, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_pts = r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    return np.vstack([outer_pts, inner_pts])

def _optimize_ring_structure(start_time, deadline):
    """
    Use Differential Evolution to find the best 2-ring (or similar) structure parameters.
    This provides a strong global starting point.
    """
    best_score = 0.0
    best_points = None
    
    # Try a few different splits
    splits = [(7, 7), (8, 6), (9, 5)]
    
    for n_out, n_in in splits:
        g = np.gcd(n_out, n_in)
        theta_max = 2 * np.pi / g
        
        def objective(params):
            r, theta = params
            if r < 0.1 or r > 0.95: # Keep points reasonably spread
                return 0.0
            pts = _generate_ring_config(n_out, n_in, r, theta)
            return -_score(pts)
        
        bounds = [(0.1, 0.95), (0.0, theta_max)]
        
        # Use DE with limited iterations to stay within time budget
        # We allocate roughly 10-15 seconds per split
        time_per_split = min(15.0, (deadline - start_time) / 3.0 - 1.0)
        
        # Estimate maxiter based on time
        # Each eval is ~1ms, DE overhead is higher.
        # popsize 10, maxiter 20 -> 200 evals ~ 0.2s + overhead.
        # We can afford more.
        maxiter = 30
        popsize = 10
        
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=maxiter, 
                popsize=popsize,
                tol=1e-5,
                polish=False # We will do our own polish
            )
            
            if res.fun < 0: # Valid score found
                r, theta = res.x
                pts = _generate_ring_config(n_out, n_in, r, theta)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
        except Exception:
            pass
            
        if time.time() > deadline - 30:
            break
            
    # If DE fails or finds nothing, fall back to a heuristic guess
    if best_points is None:
        best_points = _generate_ring_config(7, 7, 0.6, np.pi/7)
        best_score = _score(best_points)
        
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Phase 1: Global Structure Search (Ring Optimization)
    # This is the key structural insight from Inspiration 1
    best_points, best_score = _optimize_ring_structure(start_time, deadline)
    
    # Phase 2: Local Refinement
    # Use the fast local search to break symmetry and fine-tune
    if time.time() < deadline - 10:
        polished_pts, polished_s = _local_search_fast(best_points, max_iters=300, step_size=0.02)
        if polished_s > best_score:
            best_score = polished_s
            best_points = polished_pts
    
    # Phase 3: Gradient-based Polish
    # Use L-BFGS with soft-min to squeeze out more precision
    if time.time() < deadline - 5:
        final_pts, final_s = _lbfgs_polish(best_points, deadline, max_iter=300)
        if final_s > best_score:
            best_score = final_s
            best_points = final_pts
            
    # Phase 4: Final Local Search with smaller steps
    if time.time() < deadline - 1:
        final_pts, final_s = _local_search_fast(best_points, max_iters=200, step_size=0.005)
        if final_s > best_score:
            best_score = final_s
            best_points = final_pts

    return best_points
