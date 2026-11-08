"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys

def _all_triangle_areas(points):
    """Compute all triangle areas vectorized. Returns array of shape (C(n,3),)."""
    n = len(points)
    # Get all pairs of indices
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    # Area = 0.5 * |cross(pj - pi, pk - pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    areas = _all_triangle_areas(points)
    min_tri = np.min(areas)
    return min_tri / hull_area

def _soft_score(points, temperature=0.01):
    """Smooth surrogate using log-sum-exp soft-min."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    areas = _all_triangle_areas(points)
    # Soft min: -t * log(sum(exp(-a/t)))
    # For numerical stability
    max_a = np.max(areas)
    shifted = -areas / temperature
    shifted -= np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = -temperature * (log_sum + np.max(shifted))
    return soft_min / hull_area

def _regular_polygon(n, radius=1.0):
    """Generate points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _two_rings(n, inner_frac=0.5):
    """Generate points on two concentric rings."""
    n_outer = n // 2
    n_inner = n - n_outer
    
    outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_pts = inner_frac * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    return np.vstack([outer_pts, inner_pts])

def _hexagonal_packing(n):
    """Generate hexagonal packing arrangement."""
    points = []
    # Center
    points.append([0, 0])
    # First ring (6 points)
    for i in range(6):
        angle = i * np.pi / 3
        points.append([np.cos(angle), np.sin(angle)])
    # Second ring (7 points)
    for i in range(7):
        angle = i * 2 * np.pi / 7 + np.pi / 7
        points.append([2 * np.cos(angle), 2 * np.sin(angle)])
    return np.array(points[:n])

def _optimize_nelder_mead(points, deadline, use_soft=False, temperature=0.01):
    """Optimize using Nelder-Mead with time management."""
    if use_soft:
        def objective(x):
            pts = x.reshape(-1, 2)
            return -_soft_score(pts, temperature)
    else:
        def objective(x):
            pts = x.reshape(-1, 2)
            return -_score(pts)
    
    x0 = points.flatten()
    
    # Check time before starting
    if time.time() > deadline - 5:
        return points, -objective(x0)
    
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={
            'maxiter': 5000,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
    )
    
    best_points = result.x.reshape(-1, 2)
    best_score = -result.fun
    
    return best_points, best_score

def _optimize_local_fine(points, deadline, max_iters=2000, step_size=0.005):
    """Fine-tuning local optimization with small steps."""
    points = points.copy()
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    rng = np.random.RandomState(123)
    
    for it in range(max_iters):
        if time.time() > deadline - 2:
            break
        
        # Try random perturbation
        idx = rng.randint(len(points))
        delta = rng.normal(0, step_size, 2)
        new_points = points.copy()
        new_points[idx] += delta
        
        new_score = _score(new_points)
        if new_score > current_score:
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            if it % 200 == 0:
                step_size *= 0.9
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Generate diverse initial candidates
    candidates = []
    
    # Regular polygon (best known starting point)
    candidates.append(_regular_polygon(N))
    
    # Two rings with different ratios
    for frac in [0.4, 0.5, 0.6, 0.7]:
        candidates.append(_two_rings(N, frac))
    
    # Hexagonal packing
    candidates.append(_hexagonal_packing(N))
    
    # Slightly perturbed regular polygons
    rng = np.random.RandomState(42)
    for _ in range(3):
        base = _regular_polygon(N)
        perturbed = base + rng.normal(0, 0.05, (N, 2))
        candidates.append(perturbed)
    
    # Evaluate initial candidates and find the best
    best_points = None
    best_score = -1
    
    for pts in candidates:
        if time.time() > deadline - 10:
            break
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    if best_points is None:
        return _regular_polygon(N)
    
    # Phase 1: Nelder-Mead optimization with soft objective (smoother landscape)
    if time.time() < deadline - 20:
        nm_points, nm_score = _optimize_nelder_mead(
            best_points, deadline, use_soft=True, temperature=0.05
        )
        if nm_score > best_score:
            best_score = nm_score
            best_points = nm_points
    
    # Phase 2: Nelder-Mead with exact objective (refine)
    if time.time() < deadline - 15:
        nm_points, nm_score = _optimize_nelder_mead(
            best_points, deadline, use_soft=False
        )
        if nm_score > best_score:
            best_score = nm_score
            best_points = nm_points
    
    # Phase 3: Try Nelder-Mead from second-best initial candidate
    if time.time() < deadline - 15:
        # Find second-best initial candidate
        second_best_pts = None
        second_best_score = -1
        for pts in candidates:
            if np.allclose(pts, best_points, atol=0.01):
                continue
            score = _score(pts)
            if score > second_best_score:
                second_best_score = score
                second_best_pts = pts.copy()
        
        if second_best_pts is not None:
            nm_points, nm_score = _optimize_nelder_mead(
                second_best_pts, deadline, use_soft=True, temperature=0.05
            )
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_points
    
    # Phase 4: Fine-tuning with small random steps
    if time.time() < deadline - 5:
        fine_points, fine_score = _optimize_local_fine(
            best_points, deadline, max_iters=3000, step_size=0.003
        )
        if fine_score > best_score:
            best_score = fine_score
            best_points = fine_points
    
    # Phase 5: Final Nelder-Mead polish
    if time.time() < deadline - 3:
        nm_points, nm_score = _optimize_nelder_mead(
            best_points, deadline, use_soft=False
        )
        if nm_score > best_score:
            best_score = nm_score
            best_points = nm_points
    
    return best_points if best_points is not None else _regular_polygon(N)
# EVOLVE-BLOCK-END
