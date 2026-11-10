# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools


def _score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points using vectorized operations."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Vectorized triangle area calculation
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    x1, y1 = points[i]
    x2, y2 = points[j]
    x3, y3 = points[k]
    areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
    
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    
    return min_area / A_hull


def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), angle_offset=0.0):
    """Generate points of a regular n-gon."""
    angles = angle_offset + 2 * np.pi * np.arange(n) / n
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])


def _radial_perturbation(base_points, perturbations):
    """Apply radial perturbations to base points (assumed centered at origin)."""
    n = len(perturbations)
    angles = np.arctan2(base_points[:, 1], base_points[:, 0])
    radii = np.sqrt(base_points[:, 0]**2 + base_points[:, 1]**2)
    new_radii = radii + perturbations
    new_x = new_radii * np.cos(angles)
    new_y = new_radii * np.sin(angles)
    return np.column_stack([new_x, new_y])


def _smooth_min(areas, beta=100.0):
    """Smooth minimum approximation using log-sum-exp."""
    min_val = np.min(areas)
    # Use a stable log-sum-exp
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_val - np.log(np.sum(np.exp(-beta * shifted))) / beta


def _optimize_radial(base_points, max_iter=2000, time_limit=30.0):
    """Optimize radial perturbations to maximize min triangle area."""
    n = len(base_points)
    angles = np.arctan2(base_points[:, 1], base_points[:, 0])
    base_radii = np.sqrt(base_points[:, 0]**2 + base_points[:, 1]**2)
    
    # Precompute triangle index combinations
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_arr, j_arr, k_arr = i_idx[mask], j_idx[mask], k_idx[mask]
    
    start_time = time.time()
    best_perturb = np.zeros(n)
    best_score = -1.0
    
    # Simulated annealing
    current_perturb = np.zeros(n)
    current_score = -1.0
    temperature = 0.1
    cooling = 0.999
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute current score
        current_radii = base_radii + current_perturb
        points = np.column_stack([
            current_radii * np.cos(angles),
            current_radii * np.sin(angles)
        ])
        
        # Compute triangle areas
        x1, y1 = points[i_arr]
        x2, y2 = points[j_arr]
        x3, y3 = points[k_arr]
        areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
        min_area = np.min(areas)
        
        # For the score, we need A_hull too, but for optimization we can use min_area directly
        # since the hull area changes slowly. Let's use the full score for accuracy.
        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
            current_score = min_area / A_hull
        except Exception:
            current_score = 0.0
        
        if current_score > best_score:
            best_score = current_score
            best_perturb = current_perturb.copy()
        
        # Propose a move
        idx_move = np.random.randint(n)
        delta = np.random.normal(0, temperature)
        new_perturb = current_perturb.copy()
        new_perturb[idx_move] += delta
        
        # Ensure radii stay positive
        new_radii = base_radii + new_perturb
        if np.any(new_radii <= 0):
            continue
        
        new_points = np.column_stack([
            new_radii * np.cos(angles),
            new_radii * np.sin(angles)
        ])
        
        x1, y1 = new_points[i_arr]
        x2, y2 = new_points[j_arr]
        x3, y3 = new_points[k_arr]
        areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
        min_area = np.min(areas)
        
        try:
            hull = ConvexHull(new_points)
            A_hull = hull.volume
            new_score = min_area / A_hull
        except Exception:
            new_score = 0.0
        
        # Accept if better or with probability exp(-delta_score/T)
        if new_score > current_score:
            current_perturb = new_perturb
            current_score = new_score
        else:
            delta_score = current_score - new_score
            if delta_score < 0:
                current_perturb = new_perturb
                current_score = new_score
            elif np.random.random() < np.exp(-delta_score / max(temperature, 1e-10)):
                current_perturb = new_perturb
                current_score = new_score
        
        temperature *= cooling
    
    return best_perturb, best_score


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline
    
    best_points = None
    best_score = -1.0
    
    # Start with a regular 13-gon
    base_points = _regular_polygon(n, radius=1.0)
    
    # Optimize radial perturbations
    perturb, score = _optimize_radial(base_points, max_iter=5000, time_limit=400.0)
    if score > best_score:
        best_score = score
        best_points = _radial_perturbation(base_points, perturb)
    
    # Try different initial angle offsets
    for angle_offset in np.linspace(0, 2*np.pi, 8, endpoint=False):
        if time.time() > deadline - 60:
            break
        
        base_points = _regular_polygon(n, radius=1.0, angle_offset=angle_offset)
        perturb, score = _optimize_radial(base_points, max_iter=3000, time_limit=50.0)
        if score > best_score:
            best_score = score
            best_points = _radial_perturbation(base_points, perturb)
    
    # Try different base radii
    for radius in [0.8, 0.9, 1.0, 1.1, 1.2]:
        if time.time() > deadline - 60:
            break
        
        base_points = _regular_polygon(n, radius=radius)
        perturb, score = _optimize_radial(base_points, max_iter=3000, time_limit=50.0)
        if score > best_score:
            best_score = score
            best_points = _radial_perturbation(base_points, perturb)
    
    # Final local refinement on the best solution
    if best_points is not None and time.time() < deadline - 30:
        # Convert to radial representation
        angles = np.arctan2(best_points[:, 1], best_points[:, 0])
        radii = np.sqrt(best_points[:, 0]**2 + best_points[:, 1]**2)
        
        # Fine-tune each point's radius
        for _ in range(100):
            if time.time() > deadline - 30:
                break
            
            idx = np.random.randint(n)
            delta = np.random.normal(0, 0.01)
            new_radii = radii.copy()
            new_radii[idx] += delta
            
            if new_radii[idx] <= 0:
                continue
            
            new_points = np.column_stack([
                new_radii * np.cos(angles),
                new_radii * np.sin(angles)
            ])
            
            new_score = _score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
                radii = new_radii
    
    # Normalize to [0,1]^2
    if best_points is not None:
        min_x, min_y = np.min(best_points, axis=0)
        max_x, max_y = np.max(best_points, axis=0)
        width = max_x - min_x
        height = max_y - min_y
        scale = max(width, height)
        
        if scale > 0:
            best_points = (best_points - np.array([min_x, min_y])) / scale
            best_points += 0.01  # Add small margin
            best_points = np.clip(best_points, 0.01, 0.99)
    
    if best_points is None:
        best_points = _regular_polygon(n, radius=1.0)
        min_x, min_y = np.min(best_points, axis=0)
        max_x, max_y = np.max(best_points, axis=0)
        width = max_x - min_x
        height = max_y - min_y
        scale = max(width, height)
        if scale > 0:
            best_points = (best_points - np.array([min_x, min_y])) / scale
            best_points += 0.01
            best_points = np.clip(best_points, 0.01, 0.99)
    
    return best_points


# EVOLVE-BLOCK-END
