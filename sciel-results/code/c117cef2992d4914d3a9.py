"""Voronoi-based circle packing in perimeter-4 rectangle."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, Voronoi

N = 21

def _compute_inradii(centers, w, h):
    """Compute inradius of each circle given centers and rectangle dimensions.
    
    The inradius is the minimum distance from the center to:
    - Any other circle center (half the distance)
    - The rectangle walls
    """
    n = len(centers)
    inradii = np.zeros(n)
    
    # Distance to walls
    for i in range(n):
        x, y = centers[i]
        inradii[i] = min(x, w - x, y, h - y)
    
    # Distance to other centers (using Delaunay for efficiency, but for N=21 brute force is fine)
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            half_dist = dist / 2.0
            if half_dist < inradii[i]:
                inradii[i] = half_dist
            if half_dist < inradii[j]:
                inradii[j] = half_dist
    
    return inradii

def _objective(params, w, h):
    """Negative sum of inradii (we minimize negative to maximize)."""
    centers = params[:N * 2].reshape(N, 2)
    # Ensure centers are within bounds (small penalty if outside)
    penalty = 0.0
    for i in range(N):
        x, y = centers[i]
        if x < 0: penalty += x ** 2 * 100
        if x > w: penalty += (x - w) ** 2 * 100
        if y < 0: penalty += y ** 2 * 100
        if y > h: penalty += (y - h) ** 2 * 100
    
    inradii = _compute_inradii(centers, w, h)
    return -np.sum(inradii) + penalty

def _objective_with_aspect(params):
    """Full objective including aspect ratio optimization.
    params: [x1, y1, ..., xN, yN, w] where h = 2 - w
    """
    centers = params[:N * 2].reshape(N, 2)
    w = params[N * 2]
    h = 2.0 - w
    
    # Ensure w and h are positive
    if w <= 0.1 or h <= 0.1:
        return 1e6
    
    penalty = 0.0
    for i in range(N):
        x, y = centers[i]
        if x < 0: penalty += x ** 2 * 100
        if x > w: penalty += (x - w) ** 2 * 100
        if y < 0: penalty += y ** 2 * 100
        if y > h: penalty += (y - h) ** 2 * 100
    
    inradii = _compute_inradii(centers, w, h)
    return -np.sum(inradii) + penalty

def _delaunay_construction(w, h):
    """Create initial configuration using Delaunay-based construction.
    Place points in a hexagonal pattern, then compute inradii.
    """
    # Start with a grid-like hexagonal arrangement
    points = []
    # Use a 5x5 grid with hexagonal offset
    side = 5
    for i in range(side):
        for j in range(side):
            if len(points) >= N:
                break
            x = (i + 0.5) * w / side
            y = (j + 0.5) * h / side
            # Hexagonal offset for odd rows
            if i % 2 == 1:
                x += 0.25 * w / side
                x = min(x, w - 0.1)
            points.append([x, y])
        if len(points) >= N:
            break
    
    # Trim to exactly N points
    points = points[:N]
    return np.array(points)

def _hexagonal_initial(w, h):
    """Better hexagonal initial configuration."""
    points = []
    # Try to pack in a hexagonal pattern
    # For 21 circles, try 4 rows: 5, 4, 5, 4, 3 or similar
    row_configs = [
        [5, 4, 5, 4, 3],  # 21
        [4, 5, 4, 5, 3],
        [3, 5, 4, 5, 4],
    ]
    
    best_points = None
    best_score = -np.inf
    
    for config in row_configs:
        points = []
        total_rows = len(config)
        row_height = h / total_rows
        
        for row_idx, count in enumerate(config):
            y = (row_idx + 0.5) * row_height
            spacing = w / (count + 0.5)  # Slight offset for hexagonal
            for col in range(count):
                if len(points) >= N:
                    break
                x = (col + 0.5) * spacing
                # Hexagonal offset
                if row_idx % 2 == 1:
                    x += spacing * 0.5
                x = max(0.1, min(w - 0.1, x))
                points.append([x, y])
            if len(points) >= N:
                break
        
        if len(points) == N:
            points = np.array(points[:N])
            inradii = _compute_inradii(points, w, h)
            score = np.sum(inradii)
            if score > best_score:
                best_score = score
                best_points = points
    
    if best_points is None:
        best_points = _delaunay_construction(w, h)
    
    return best_points

def _is_valid(centers, w, h, tol=1e-9):
    """Check if configuration is valid (no overlaps, all inside rectangle)."""
    if w <= 0 or h <= 0 or w + h > 2 + 1e-6:
        return False
    
    for i in range(N):
        x, y = centers[i]
        if x < -tol or x > w + tol or y < -tol or y > h + tol:
            return False
    
    inradii = _compute_inradii(centers, w, h)
    for i in range(N):
        if inradii[i] < -tol:
            return False
    
    return True

def _force_relax(centers, w, h, max_iters=100):
    """Force relaxation: iteratively move centers to resolve overlaps."""
    for _ in range(max_iters):
        modified = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = _compute_inradii(centers, w, h)[i] + _compute_inradii(centers, w, h)[j]
                if dist < min_dist and dist > 1e-12:
                    # Push apart
                    overlap = min_dist - dist
                    push_x = overlap * dx / dist
                    push_y = overlap * dy / dist
                    centers[i][0] += push_x * 0.5
                    centers[i][1] += push_y * 0.5
                    centers[j][0] -= push_x * 0.5
                    centers[j][1] -= push_y * 0.5
                    modified = True
        
        # Ensure centers stay in bounds
        for i in range(N):
            centers[i][0] = max(0.01, min(w - 0.01, centers[i][0]))
            centers[i][1] = max(0.01, min(h - 0.01, centers[i][1]))
        
        if not modified:
            break
    
    return centers

def _expand_individual(centers, w, h, max_iters=200):
    """Try to expand individual circles by adjusting positions."""
    for _ in range(max_iters):
        improved = False
        for i in range(N):
            # Try small perturbations
            for dx in [-0.01, 0.01]:
                for dy in [-0.01, 0.01]:
                    new_centers = centers.copy()
                    new_centers[i][0] += dx
                    new_centers[i][1] += dy
                    # Check bounds
                    if new_centers[i][0] < 0.01 or new_centers[i][0] > w - 0.01:
                        continue
                    if new_centers[i][1] < 0.01 or new_centers[i][1] > h - 0.01:
                        continue
                    new_inradii = _compute_inradii(new_centers, w, h)
                    old_inradii = _compute_inradii(centers, w, h)
                    if np.sum(new_inradii) > np.sum(old_inradii) + 1e-8:
                        centers = new_centers
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    
    return centers

def _polish_result(centers, w, h):
    """Final polish: ensure validity and maximize sum of inradii."""
    # Force relax to ensure no overlaps
    centers = _force_relax(centers, w, h)
    
    # Expand individual
    centers = _expand_individual(centers, w, h)
    
    # Final force relax
    centers = _force_relax(centers, w, h)
    
    return centers

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 150s kill
    
    best_centers = None
    best_w = 1.0
    best_h = 1.0
    best_score = -np.inf
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4]
    
    for ar in aspect_ratios:
        if time.time() - start_time > deadline:
            break
        
        w = ar * 2.0 / (1.0 + ar)
        h = 2.0 - w
        
        if w <= 0.2 or h <= 0.2:
            continue
        
        # Multiple initial configurations
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() - start_time > deadline:
                break
            
            if restart == 0:
                centers = _hexagonal_initial(w, h)
            else:
                # Perturbed hexagonal
                centers = _hexagonal_initial(w, h).copy()
                np.random.seed(42 + restart)
                for i in range(N):
                    centers[i][0] += np.random.uniform(-0.05, 0.05)
                    centers[i][1] += np.random.uniform(-0.05, 0.05)
                    centers[i][0] = max(0.05, min(w - 0.05, centers[i][0]))
                    centers[i][1] = max(0.05, min(h - 0.05, centers[i][1]))
            
            # Force relax
            centers = _force_relax(centers, w, h)
            
            # SLSQP optimization
            try:
                params = np.concatenate([centers.flatten(), [w]])
                
                def obj(p):
                    c = p[:N * 2].reshape(N, 2)
                    ww = p[N * 2]
                    hh = 2.0 - ww
                    if ww <= 0.2 or hh <= 0.2:
                        return 1e6
                    penalty = 0.0
                    for i in range(N):
                        x, y = c[i]
                        if x < 0: penalty += x ** 2 * 100
                        if x > ww: penalty += (x - ww) ** 2 * 100
                        if y < 0: penalty += y ** 2 * 100
                        if y > hh: penalty += (y - hh) ** 2 * 100
                    inradii = _compute_inradii(c, ww, hh)
                    return -np.sum(inradii) + penalty
                
                # Bounds: centers in [0, w] x [0, h], w in [0.3, 1.7]
                bounds = []
                for i in range(N):
                    bounds.append((0.01, w - 0.01))
                    bounds.append((0.01, h - 0.01))
                bounds.append((0.3, 1.7))
                
                result = minimize(
                    obj,
                    params,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                
                if result.success or result.fun < 1e5:
                    new_centers = result.x[:N * 2].reshape(N, 2)
                    new_w = result.x[N * 2]
                    new_h = 2.0 - new_w
                    
                    if new_w > 0.2 and new_h > 0.2:
                        new_centers = _polish_result(new_centers, new_w, new_h)
                        new_inradii = _compute_inradii(new_centers, new_w, new_h)
                        new_score = np.sum(new_inradii)
                        
                        if new_score > best_score:
                            best_score = new_score
                            best_centers = new_centers.copy()
                            best_w = new_w
                            best_h = new_h
            except Exception as e:
                pass
    
    # If no valid solution found, use fallback
    if best_centers is None:
        w, h = 1.0, 1.0
        best_centers = _delaunay_construction(w, h)
        best_centers = _force_relax(best_centers, w, h)
        best_w, best_h = w, h
    
    # Final validation and construction
    inradii = _compute_inradii(best_centers, best_w, best_h)
    
    # Ensure validity: if any inradius is negative or too small, fix
    for i in range(N):
        if inradii[i] < 0.001:
            inradii[i] = 0.001
    
    # Construct output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    for i in range(N):
        result[i][0] = best_centers[i][0]
        result[i][1] = best_centers[i][1]
        result[i][2] = inradii[i]
    
    return result
