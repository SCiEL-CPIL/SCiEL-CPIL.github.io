"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


def _min_triangle_area_fast(points):
    """
    Fast vectorized computation of the minimum triangle area.
    points: (N, 2) array.
    Returns: min area (float).
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            d_ij = pj - pi
            diffs = points - pi
            crosses = d_ij[0] * diffs[:, 1] - d_ij[1] * diffs[:, 0]
            areas = 0.5 * np.abs(crosses)
            areas[i] = float('inf')
            areas[j] = float('inf')
            current_min = np.min(areas)
            if current_min < min_area:
                min_area = current_min
    return min_area


def _is_inside_triangle(points):
    """
    Check if all points are inside the unit equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    Returns: True if all points are inside (with small tolerance).
    """
    x = points[:, 0]
    y = points[:, 1]
    tol = 1e-9
    
    # y >= 0
    if np.any(y < -tol):
        return False
    
    # y <= sqrt(3)*x (left edge)
    if np.any(y > np.sqrt(3) * x + tol):
        return False
    
    # y <= sqrt(3)*(1-x) (right edge)
    if np.any(y > np.sqrt(3) * (1 - x) + tol):
        return False
    
    # x >= 0, x <= 1
    if np.any(x < -tol) or np.any(x > 1 + tol):
        return False
    
    return True


def _clip_to_triangle(points):
    """
    Clip points to be inside the triangle.
    """
    result = points.copy()
    for i in range(len(result)):
        x, y = result[i]
        
        # Clip to bounding box first
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(np.sqrt(3)/2, y))
        
        # Left edge: y <= sqrt(3)*x
        if y > np.sqrt(3) * x:
            # Project to line y = sqrt(3)*x
            # Closest point: minimize (x'-x)^2 + (y'-y)^2 subject to y' = sqrt(3)*x'
            # Lagrangian: 2(x'-x) + lambda*sqrt(3) = 0, 2(y'-y) + lambda = 0
            # From second: lambda = -2(y'-y)
            # From first: x' - x = -lambda*sqrt(3)/2 = (y'-y)*sqrt(3)
            # So x' = x + sqrt(3)*(y'-y), and y' = sqrt(3)*x'
            # y' = sqrt(3)*(x + sqrt(3)*(y'-y)) = sqrt(3)*x + 3*(y'-y)
            # y' = sqrt(3)*x + 3*y' - 3*y
            # 3*y - y' = sqrt(3)*x + 3*y' - y'  -- wait, let me redo
            # y' = sqrt(3)*x + 3(y' - y)
            # y' = sqrt(3)*x + 3y' - 3y
            # 3y - sqrt(3)*x = 2y'
            # y' = (3y - sqrt(3)*x) / 2
            # x' = y' / sqrt(3)
            y_new = (3 * y - np.sqrt(3) * x) / 2.0
            x_new = y_new / np.sqrt(3)
            # But this might not be the closest point if the projection is outside the segment
            # For now, use this projection
            x, y = x_new, y_new
        
        # Right edge: y <= sqrt(3)*(1-x)
        if y > np.sqrt(3) * (1 - x):
            # Line: y = sqrt(3) - sqrt(3)*x, or sqrt(3)*x + y = sqrt(3)
            # Minimize (x'-x)^2 + (y'-y)^2 subject to sqrt(3)*x' + y' = sqrt(3)
            # Lagrangian: 2(x'-x) + lambda*sqrt(3) = 0, 2(y'-y) + lambda = 0
            # lambda = -2(y'-y)
            # x' - x = (y'-y)*sqrt(3)
            # x' = x + sqrt(3)*(y'-y)
            # sqrt(3)*(x + sqrt(3)*(y'-y)) + y' = sqrt(3)
            # sqrt(3)*x + 3(y'-y) + y' = sqrt(3)
            # sqrt(3)*x + 4y' - 3y = sqrt(3)
            # 4y' = sqrt(3) - sqrt(3)*x + 3y
            # y' = (sqrt(3)*(1-x) + 3y) / 4
            y_new = (np.sqrt(3) * (1 - x) + 3 * y) / 4.0
            x_new = (np.sqrt(3) - y_new) / np.sqrt(3)
            x, y = x_new, y_new
        
        # Final clip to bounding box
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(np.sqrt(3)/2, y))
        
        result[i] = [x, y]
    
    return result


def _objective_flat(flat_points):
    """
    Objective function for optimization.
    Returns: negative min area (to maximize) + penalty for being outside.
    """
    points = flat_points.reshape(N, 2)
    
    # Check if inside
    if not _is_inside_triangle(points):
        # Compute penalty
        x = points[:, 0]
        y = points[:, 1]
        pen = 0.0
        # y < 0
        pen += np.sum(np.clip(-y, 0, None))
        # y > sqrt(3)*x
        pen += np.sum(np.clip(y - np.sqrt(3) * x, 0, None))
        # y > sqrt(3)*(1-x)
        pen += np.sum(np.clip(y - np.sqrt(3) * (1 - x), 0, None))
        # x < 0 or x > 1
        pen += np.sum(np.clip(-x, 0, None)) + np.sum(np.clip(x - 1, 0, None))
        return pen * 1000.0
    
    min_area = _min_triangle_area_fast(points)
    return -min_area


def _generate_structured_inits():
    """
    Generate diverse structured initial points.
    Returns a list of arrays, each of shape (11, 2).
    """
    inits = []
    
    # Helper to convert barycentric (11, 3) to cartesian (11, 2)
    def bary_to_cart(bary):
        # bary: (N, 3) where cols are weights for (0,0), (1,0), (0.5, sqrt(3)/2)
        # x = 0*w0 + 1*w1 + 0.5*w2 = w1 + 0.5*w2
        # y = 0*w0 + 0*w1 + sqrt(3)/2*w2 = sqrt(3)/2 * w2
        x = bary[:, 1] + 0.5 * bary[:, 2]
        y = np.sqrt(3)/2 * bary[:, 2]
        return np.column_stack((x, y))

    # 1. "Hexagonal" like packing adapted for 11 points
    # Vertices (3) + Edge midpoints (3) + Inner hexagon (5) ? 
    # Let's try a symmetric 11-point configuration.
    # 3 vertices
    # 3 points on edges (1/3, 2/3 along edges)
    # 3 points forming a small triangle in center
    # 2 more points?
    
    # Configuration A: 3 vertices + 3 edge points + 5 center points
    bary_verts = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Edge points at 1/3 and 2/3? Let's use 1/3 from vertices for better spread.
    # Edge 0-1: (2/3, 1/3, 0), (1/3, 2/3, 0) -> Let's just take one per edge to save space for center
    bary_edges = np.array([
        [2.0/3, 1.0/3, 0.0], # On edge 0-1
        [1.0/3, 0.0, 2.0/3], # On edge 0-2
        [0.0, 2.0/3, 1.0/3]  # On edge 1-2
    ])
    
    # Center 5 points: Center + 4 around it
    center = np.array([1.0/3, 1.0/3, 1.0/3])
    bary_center_5 = [center]
    # Add 4 points around center in cartesian space, then convert back?
    # Easier: define in cartesian, check if inside, then optimize.
    # Let's just generate a grid of barycentric points and pick 5 good ones.
    # Or simply: Center + 4 points at distance r from center in cartesian.
    
    pts_cart = np.vstack([bary_to_cart(bary_verts), bary_to_cart(bary_edges)])
    
    # Add center
    pts_cart = np.vstack([pts_cart, bary_to_cart(center.reshape(1,3))])
    
    # Add 4 more points around center
    c_x, c_y = pts_cart[-1]
    for k in range(4):
        a = 2 * np.pi * k / 4 + np.pi/4 # 45, 135, 225, 315
        r = 0.15
        nx = c_x + r * np.cos(a)
        ny = c_y + r * np.sin(a)
        pts_cart = np.vstack([pts_cart, [[nx, ny]]])
        
    # We have 3+3+1+4 = 11 points.
    # Clip to triangle to ensure validity
    pts_cart = _clip_to_triangle(pts_cart)
    inits.append(pts_cart)
    
    # 2. Random barycentric starts (5 diverse ones)
    rng = np.random.default_rng(42)
    for _ in range(5):
        bary = rng.random((N, 3))
        bary /= bary.sum(axis=1, keepdims=True)
        pts = bary_to_cart(bary)
        pts = _clip_to_triangle(pts)
        inits.append(pts)
        
    # 3. A "grid" like start
    # 4x4 grid in barycentric, pick 11
    grid_bary = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if i+j+k == 3:
                    grid_bary.append([i/3.0, j/3.0, k/3.0])
    grid_bary = np.array(grid_bary)
    # grid_bary has 10 points (1,3,6? No. i+j+k=3. 
    # i=0: (0,3,0), (0,2,1), (0,1,2), (0,0,3) -> 4
    # i=1: (1,2,0), (1,1,1), (1,0,2) -> 3
    # i=2: (2,1,0), (2,0,1) -> 2
    # i=3: (3,0,0) -> 1
    # Total 10 points. Need 11.
    # Add center to make 11.
    grid_bary = np.vstack([grid_bary, [1/3, 1/3, 1/3]])
    pts_grid = bary_to_cart(grid_bary)
    pts_grid = _clip_to_triangle(pts_grid)
    inits.append(pts_grid)
    
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # Generate structured inits
    inits = _generate_structured_inits()
    
    # Evaluate each init
    for init in inits:
        pts = np.array(init, dtype=float).reshape(N, 2)
        if _is_inside_triangle(pts):
            score = _min_triangle_area_fast(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
    
    # If no valid init, use random
    if best_points is None:
        rng = np.random.default_rng(42)
        bary = rng.random((N, 3))
        bary /= bary.sum(axis=1, keepdims=True)
        x = bary[:, 1] + 0.5 * bary[:, 2]
        y = np.sqrt(3)/2 * bary[:, 2]
        best_points = np.column_stack((x, y))
        best_score = _min_triangle_area_fast(best_points)
    
    # Run differential evolution
    bounds = [(0, 1), (0, np.sqrt(3)/2)] * N
    
    try:
        result = differential_evolution(
            _objective_flat,
            bounds,
            maxiter=500,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False,
            init='sobol',
            x0=best_points.flatten(),
        )
        
        if result.fun < 0:
            pts = result.x.reshape(N, 2)
            if _is_inside_triangle(pts):
                score = _min_triangle_area_fast(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
    except Exception:
        pass
    
    # Polish with Nelder-Mead from best point
    if time.time() < deadline - 10:
        try:
            nm_result = minimize(
                _objective_flat,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if nm_result.fun < 0:
                pts = nm_result.x.reshape(N, 2)
                if _is_inside_triangle(pts):
                    score = _min_triangle_area_fast(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
        except Exception:
            pass
    
    # Multi-start NM from diverse points
    if time.time() < deadline - 20:
        rng = np.random.default_rng(123)
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            
            # Random perturbation of best point
            perturb = rng.normal(0, 0.05, size=(N, 2))
            pts = best_points + perturb
            pts = _clip_to_triangle(pts)
            
            if not _is_inside_triangle(pts):
                continue
            
            try:
                nm_res = minimize(
                    _objective_flat,
                    pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                
                if nm_res.fun < 0:
                    pts2 = nm_res.x.reshape(N, 2)
                    if _is_inside_triangle(pts2):
                        score = _min_triangle_area_fast(pts2)
                        if score > best_score:
                            best_score = score
                            best_points = pts2.copy()
            except Exception:
                pass
    
    # Final safety check
    best_points = _clip_to_triangle(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
