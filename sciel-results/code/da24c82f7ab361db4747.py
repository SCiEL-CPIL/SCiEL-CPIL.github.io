"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
NUM_TRIPLES = N * (N - 1) * (N - 2) // 6

# Precompute triple indices
_triple_indices = np.array(list(combinations(range(N), 3)))
_i_idx = _triple_indices[:, 0]
_j_idx = _triple_indices[:, 1]
_k_idx = _triple_indices[:, 2]

# EVOLVE-BLOCK-START

def _compute_hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        if np.allclose(pts[:, 0], pts[0, 0]) or np.allclose(pts[:, 1], pts[0, 1]):
            return 0.0
        hull = ConvexHull(pts)
        area = hull.volume
        if area < 1e-9:
            return 0.0
        return area
    except Exception:
        return 0.0

def _compute_min_ratio(pts):
    """
    Compute the minimum triangle area divided by the convex hull area.
    Returns the score (higher is better) or 0 if invalid.
    """
    hull_area = _compute_hull_area(pts)
    if hull_area == 0.0:
        return 0.0
    
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _params_to_points(params):
    """Convert 22 free coordinates to full point array."""
    pts = np.empty((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = params.reshape(N - 2, 2)
    return pts

def _score_from_params(params):
    """Compute score directly from params."""
    pts = _params_to_points(params)
    return _compute_min_ratio(pts)

def _generate_concentric_ring_init(rng, num_rings=3):
    """
    Generate initial points using concentric rings with angular offsets.
    This provides good initial spread for the Heilbronn problem.
    """
    pts = np.zeros((N, 2))
    
    # Distribute points across rings
    # Ring 0: center (1 point)
    # Ring 1: 4 points
    # Ring 2: 5 points
    # Ring 3: 3 points
    ring_sizes = [1, 4, 5, 3]
    ring_radii = [0.0, 0.35, 0.65, 0.95]
    
    idx = 0
    for ring in range(num_rings):
        size = ring_sizes[ring]
        radius = ring_radii[ring]
        
        if ring == 0:
            # Center point
            pts[idx] = [0.0, 0.0]
            idx += 1
        else:
            # Points on this ring
            base_angle = rng.uniform(0, 2 * np.pi)
            for k in range(size):
                angle = base_angle + 2 * np.pi * k / size
                r = radius * (1.0 + rng.normal(0, 0.05))
                pts[idx] = [r * np.cos(angle), r * np.sin(angle)]
                idx += 1
    
    # Add small random perturbation
    pts += rng.normal(0, 0.02, (N, 2))
    
    # Transform so p[0] is at (0,0) and p[1] is at (1,0)
    pts = pts - pts[0]
    v = pts[1]
    length = np.linalg.norm(v)
    if length < 1e-9:
        return rng.uniform(-1, 1, 22)
    pts = pts / length
    theta = np.arctan2(pts[1, 1], pts[1, 0])
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    
    return pts[2:].flatten()

def _generate_regular_polygon_init(rng):
    """Generate initial points from regular polygon with perturbation."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles += rng.uniform(-0.1, 0.1, N)
    radii = 1.0 + rng.normal(0, 0.05, N)
    
    pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    
    # Transform
    pts = pts - pts[0]
    v = pts[1]
    length = np.linalg.norm(v)
    if length < 1e-9:
        return rng.uniform(-1, 1, 22)
    pts = pts / length
    theta = np.arctan2(pts[1, 1], pts[1, 0])
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    
    return pts[2:].flatten()

def _generate_halton_init(rng):
    """Generate initial points from Halton sequence with perturbation."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = np.zeros((N, 2))
    for i in range(N):
        pts[i, 0] = halton(i + 1, 2)
        pts[i, 1] = halton(i + 1, 3)
    
    pts += rng.normal(0, 0.05, (N, 2))
    
    # Transform
    pts = pts - pts[0]
    v = pts[1]
    length = np.linalg.norm(v)
    if length < 1e-9:
        return rng.uniform(-1, 1, 22)
    pts = pts / length
    theta = np.arctan2(pts[1, 1], pts[1, 0])
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    
    return pts[2:].flatten()

def _adaptive_local_search(params, best_score, rng, time_limit, start_time):
    """
    Direct local search with adaptive step sizes on the exact objective.
    Uses coordinate-wise perturbations with adaptive step size.
    """
    current_params = params.copy()
    current_score = _score_from_params(current_params)
    
    # Initial step size
    step_size = 0.05
    min_step = 1e-6
    max_iter = 5000
    
    for it in range(max_iter):
        if (time.time() - start_time) > time_limit - 2.0:
            break
        
        # Choose a random coordinate to perturb
        dim = 2 * (N - 2)  # 22 dimensions
        coord = rng.integers(0, dim)
        
        # Try positive and negative perturbations
        best_new_params = None
        best_new_score = current_score
        
        for direction in [1, -1]:
            new_params = current_params.copy()
            new_params[coord] += direction * step_size
            
            new_score = _score_from_params(new_params)
            
            if new_score > best_new_score:
                best_new_score = new_score
                best_new_params = new_params
        
        if best_new_params is not None:
            # Accept the improvement
            current_params = best_new_params
            current_score = best_new_score
            
            if current_score > best_score[0]:
                best_score[0] = current_score
                best_score[1] = current_params.copy()
            
            # Increase step size slightly on improvement
            step_size = min(step_size * 1.1, 0.1)
        else:
            # No improvement, decrease step size
            step_size *= 0.8
            if step_size < min_step:
                break
    
    return current_params, current_score

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(12345)
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Generate diverse starting points
    starts = []
    
    # Concentric ring initialization (multiple variants)
    for _ in range(5):
        starts.append(_generate_concentric_ring_init(rng))
    
    # Regular polygon initialization
    for _ in range(3):
        starts.append(_generate_regular_polygon_init(rng))
    
    # Halton initialization
    for _ in range(3):
        starts.append(_generate_halton_init(rng))
    
    # Random initialization
    for _ in range(2):
        starts.append(rng.uniform(-1, 1, 22))
    
    # Phase 2: For each start, run adaptive local search
    for i, start_params in enumerate(starts):
        if (time.time() - start_time) > time_limit - 5.0:
            break
        
        # Evaluate initial score
        initial_score = _score_from_params(start_params)
        if initial_score > best_score:
            best_score = initial_score
            best_params = start_params.copy()
        
        # Run adaptive local search
        score_tracker = [best_score, best_params]
        try:
            optimized_params, final_score = _adaptive_local_search(
                start_params, score_tracker, rng, time_limit, start_time
            )
        except Exception:
            continue
    
    # Phase 3: Polish top candidates with Nelder-Mead if time permits
    if (time.time() - start_time) < time_limit - 10.0 and best_params is not None:
        try:
            res = minimize(
                lambda p: -_score_from_params(p),
                best_params,
                method='Nelder-Mead',
                options={'maxfev': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            new_score = _score_from_params(res.x)
            if new_score > best_score:
                best_score = new_score
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Construct final points
    if best_params is None:
        best_params = _generate_concentric_ring_init(rng)
    
    final_pts = _params_to_points(best_params)
    return final_pts

# EVOLVE-BLOCK-END
