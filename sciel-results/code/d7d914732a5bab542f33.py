import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13

# EVOLVE-BLOCK-START

# Precompute all triangle index triples for vectorized area calculation
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)
_NUM_TRIPLES = len(_TRIPLE_IDX)


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


def _soft_min_objective(x_flat, beta=80.0, deadline=None):
    """
    Smooth surrogate for -min(triangle_area)/hull_area.
    Uses log-sum-exp to create a differentiable approximation of the min.
    """
    if deadline is not None and time.time() > deadline:
        return 1e10
    
    points = x_flat.reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    a_min_val = np.min(areas)
    if a_min_val < 1e-15:
        return 1e10
    
    ratios = areas / hull_area
    r_min = np.min(ratios)
    if r_min < 1e-15:
        return 1e10
    
    shifted = -beta * (ratios - r_min)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = r_min + log_sum / beta
    return -soft_min_val


def _hard_objective(x_flat, deadline=None):
    """Hard objective: -min_triangle_area / hull_area"""
    if deadline is not None and time.time() > deadline:
        return 1e10
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _normalize(points):
    """Normalize points to have hull area 1 and center at origin."""
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return points
    scale = 1.0 / np.sqrt(hull_area)
    center = np.mean(points, axis=0)
    return (points - center) * scale


def _simulated_annealing(points, rng, deadline, t0=0.5, t1=1e-6, alpha=0.9998, step_scale=0.15, min_step=1e-8):
    """
    Simulated Annealing to maximize min triangle area / hull area.
    Uses vectorized area computation for fast evaluation.
    """
    current_points = points.copy()
    current_score = _score(current_points)
    if current_score <= 0:
        current_score = 1e-10
    
    best_points = current_points.copy()
    best_score = current_score
    
    temperature = t0
    step = step_scale
    
    while time.time() < deadline - 2.0:
        # Generate perturbation
        noise = rng.normal(0, step, size=(N, 2))
        new_points = current_points + noise
        
        # Normalize to keep in reasonable range
        new_points = _normalize(new_points)
        
        new_score = _score(new_points)
        if new_score <= 0:
            new_score = 1e-10
        
        delta = new_score - current_score
        
        if delta > 0:
            # Accept improvement
            current_points = new_points
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        else:
            # Accept with probability exp(delta / temperature)
            prob = np.exp(delta / temperature)
            if rng.random() < prob:
                current_points = new_points
                current_score = new_score
        
        # Cool down
        temperature *= alpha
        step = max(min_step, step * 0.99995)
        
        if temperature < t1:
            break
    
    return best_points, best_score


def _local_refinement(points, deadline, max_iter=300):
    """
    Local refinement using Nelder-Mead on the hard objective.
    """
    try:
        result = minimize(
            _hard_objective,
            points.flatten(),
            args=(deadline,),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        new_points = result.x.reshape(N, 2)
        new_points = _normalize(new_points)
        new_score = _score(new_points)
        return new_points, new_score
    except Exception:
        return points, _score(points)


def _build_symmetric_config(k_inner, k_outer, r_inner, r_outer, phase_offset=0.0):
    """
    Build a configuration with k_inner points on an inner circle and k_outer on an outer circle.
    Total points = k_inner + k_outer.
    """
    total = k_inner + k_outer
    if total != N:
        raise ValueError(f"Total points {total} != {N}")
    
    points = np.zeros((N, 2))
    
    # Inner points
    angles_in = np.linspace(0, 2 * np.pi, k_inner, endpoint=False) + phase_offset
    points[:k_inner, 0] = r_inner * np.cos(angles_in)
    points[:k_inner, 1] = r_inner * np.sin(angles_in)
    
    # Outer points
    angles_out = np.linspace(0, 2 * np.pi, k_outer, endpoint=False) + phase_offset
    points[k_inner:, 0] = r_outer * np.cos(angles_out)
    points[k_inner:, 1] = r_outer * np.sin(angles_out)
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_points = None
    best_score = 0.0
    
    rng = np.random.default_rng(42)
    
    # Initial candidates: regular polygon and some symmetric configurations
    candidates = []
    
    # Regular 13-gon
    init_points = _regular_polygon(N)
    init_points = _normalize(init_points)
    candidates.append(init_points)
    
    # Symmetric configurations: (inner, outer, r_inner, r_outer)
    # 13 = 1+12, 2+11, 3+10, 4+9, 5+8, 6+7
    symmetric_configs = [
        (1, 12, 0.3, 1.0),
        (2, 11, 0.4, 1.0),
        (3, 10, 0.5, 1.0),
        (4, 9, 0.5, 1.0),
        (5, 8, 0.6, 1.0),
        (6, 7, 0.6, 1.0),
        (5, 8, 0.4, 1.0),
        (6, 7, 0.4, 1.0),
    ]
    
    for k_in, k_out, r_in, r_out in symmetric_configs:
        try:
            pts = _build_symmetric_config(k_in, k_out, r_in, r_out)
            pts = _normalize(pts)
            candidates.append(pts)
        except Exception:
            pass
    
    # Evaluate all candidates
    for cand in candidates:
        score = _score(cand)
        if score > best_score:
            best_score = score
            best_points = cand.copy()
    
    # Phase 1: Simulated Annealing from the best candidate
    if best_points is not None:
        sa_points, sa_score = _simulated_annealing(
            best_points, rng, deadline,
            t0=0.5, t1=1e-6, alpha=0.9998, step_scale=0.15
        )
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
    
    # Local refinement on the SA result
    if time.time() < deadline - 10:
        lr_points, lr_score = _local_refinement(
            best_points, deadline, max_iter=300
        )
        if lr_score > best_score:
            best_score = lr_score
            best_points = lr_points.copy()
    
    # Phase 2: Try SA from other good candidates
    for cand in candidates[1:4]:  # Try a few more starting points
        if time.time() > deadline - 25:
            break
        
        sa_points, sa_score = _simulated_annealing(
            cand, rng, deadline,
            t0=0.3, t1=1e-5, alpha=0.9999, step_scale=0.1
        )
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
        
        # Quick local refinement
        if time.time() < deadline - 10:
            lr_points, lr_score = _local_refinement(
                best_points, deadline, max_iter=200
            )
            if lr_score > best_score:
                best_score = lr_score
                best_points = lr_points.copy()
    
    # Phase 3: Final fine-tuning
    if best_points is not None and time.time() < deadline - 5:
        lr_points, lr_score = _local_refinement(
            best_points, deadline, max_iter=300
        )
        if lr_score > best_score:
            best_score = lr_score
            best_points = lr_points.copy()
    
    if best_points is None:
        return _regular_polygon(N)
        
    return best_points

# EVOLVE-BLOCK-END
