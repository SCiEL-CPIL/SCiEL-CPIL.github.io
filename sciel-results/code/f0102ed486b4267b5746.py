import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

_TRIPLE_INDICES = _get_triple_indices(N)
_T = len(_TRIPLE_INDICES)

# Precompute triple index arrays for fast vectorized area computation
_I = _TRIPLE_INDICES[:, 0]
_J = _TRIPLE_INDICES[:, 1]
_K = _TRIPLE_INDICES[:, 2]

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _min_triangle_area(points):
    """Compute min triangle area from 14 points (vectorized)."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _score(points):
    """Compute score: min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _soft_min_score(points, sharpness=50.0):
    """
    Smooth approximation of min triangle area / hull area.
    Uses softmax over negative areas to create a differentiable objective.
    Returns a value that is close to the true min area / hull area.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    # For the soft version, we use the actual min but add a small smoothing
    # via considering the bottom few areas
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: use log-sum-exp style
    # soft_min = -log(sum(exp(-areas/sharpness))) * sharpness / _T
    # But for simplicity and speed, just use the min directly
    # The gradient will come from the optimization method
    return min_area / hull_area

def _neg_score(points):
    """Negative score for minimization."""
    return -_score(points)

def _generate_initial_points(rng, mode='random'):
    """Generate initial point configurations."""
    if mode == 'regular':
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    elif mode == 'two_ring':
        # 7 points on outer circle, 7 on inner circle
        r = rng.uniform(0.3, 0.8)
        theta_off = rng.uniform(0, 2*np.pi/7)
        angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
        angles_inner = theta_off + np.linspace(0, 2*np.pi, 7, endpoint=False)
        outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        inner_pts = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
        return np.vstack([outer_pts, inner_pts])
    elif mode == 'jittered_regular':
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = 1.0 + rng.normal(0, 0.05, N)
        angle_noise = rng.normal(0, 0.05, N)
        pts = np.column_stack([radii * np.cos(angles + angle_noise),
                               radii * np.sin(angles + angle_noise)])
        return pts
    else:
        # Random points in unit disk
        angles = rng.uniform(0, 2*np.pi, N)
        radii = np.sqrt(rng.uniform(0.1, 1.0, N))
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _local_polish(points, deadline, rng, max_iterations=500):
    """
    Local coordinate descent polish with fast area updates.
    Moves each point slightly to improve the score.
    """
    current = points.copy()
    current_score = _score(current)
    step_size = 0.05
    best_score = current_score
    best_points = current.copy()
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            # Try small perturbations of point i
            best_local_score = current_score
            best_local_pt = current[i].copy()
            
            for dx in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                for dy in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    trial = current.copy()
                    trial[i, 0] += dx
                    trial[i, 1] += dy
                    s = _score(trial)
                    if s > best_local_score:
                        best_local_score = s
                        best_local_pt = trial[i].copy()
            
            if best_local_score > current_score + 1e-10:
                current[i] = best_local_pt
                current_score = best_local_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_points = current.copy()
        
        if not improved:
            # Reduce step size
            step_size *= 0.7
            if step_size < 1e-5:
                break
    
    return best_points, best_score

def _coordinate_descent_polish(points, deadline, rng, max_rounds=200):
    """
    More aggressive coordinate descent with random perturbations.
    """
    current = points.copy()
    current_score = _score(current)
    best_score = current_score
    best_points = current.copy()
    step_size = 0.1
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            # Random perturbations
            best_local_score = current_score
            best_local_pt = current[i].copy()
            
            for _ in range(8):
                dx = rng.uniform(-step_size, step_size)
                dy = rng.uniform(-step_size, step_size)
                trial = current.copy()
                trial[i, 0] += dx
                trial[i, 1] += dy
                s = _score(trial)
                if s > best_local_score:
                    best_local_score = s
                    best_local_pt = trial[i].copy()
            
            if best_local_score > current_score + 1e-10:
                current[i] = best_local_pt
                current_score = best_local_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_points = current.copy()
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-4:
                break
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    rng = np.random.RandomState(42)
    
    best_score = 0
    best_points = None
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Regular 14-gon
    init_reg = _generate_initial_points(rng, 'regular')
    initial_configs.append(init_reg)
    
    # Two-ring configurations with different parameters
    for r in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for theta in [0, np.pi/7, 2*np.pi/7, np.pi/14]:
            rng2 = np.random.RandomState(int(r*100) + int(theta*10))
            cfg = _generate_initial_points(rng2, 'two_ring')
            # Override with specific r and theta
            angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
            angles_inner = theta + np.linspace(0, 2*np.pi, 7, endpoint=False)
            outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
            inner_pts = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
            cfg = np.vstack([outer_pts, inner_pts])
            initial_configs.append(cfg)
    
    # Jittered regular
    for _ in range(5):
        initial_configs.append(_generate_initial_points(rng, 'jittered_regular'))
    
    # Random points
    for _ in range(5):
        initial_configs.append(_generate_initial_points(rng, 'random'))
    
    # Score all initial configurations and keep the best few
    scored_configs = []
    for cfg in initial_configs:
        s = _score(cfg)
        scored_configs.append((s, cfg))
    
    scored_configs.sort(key=lambda x: -x[0])
    
    # Polish the top configurations
    top_k = min(5, len(scored_configs))
    polished = []
    
    for s, cfg in scored_configs[:top_k]:
        if time.time() > deadline - 20:
            break
        
        # First pass: coordinate descent
        p1, s1 = _coordinate_descent_polish(cfg, deadline - 10, rng, max_rounds=100)
        if s1 > best_score:
            best_score = s1
            best_points = p1.copy()
        
        # Second pass: finer local polish
        if time.time() > deadline - 10:
            break
        p2, s2 = _local_polish(p1, deadline - 5, rng, max_iterations=200)
        if s2 > best_score:
            best_score = s2
            best_points = p2.copy()
    
    # If time permits, try scipy optimization from the best point
    if best_points is not None and time.time() > deadline - 30:
        pass  # Skip scipy if not enough time
    
    # Final check: ensure best_points is valid
    if best_points is None:
        best_points = init_reg
    
    return best_points

# EVOLVE-BLOCK-START
# Additional helper functions can be placed here if needed.
# EVOLVE-BLOCK-END
