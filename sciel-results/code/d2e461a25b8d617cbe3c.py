"""SDP relaxation approach for Heilbronn triangles."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay
import itertools
import time

def _triangle_area(p1, p2, p3):
    """Compute area of triangle given three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _is_inside_triangle(p, tri):
    """Check if point p is inside or on the boundary of triangle tri (3 points)."""
    p = np.array(p)
    a, b, c = [np.array(v) for v in tri]
    
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    
    has_neg = (d1 < -1e-9) or (d2 < -1e-9) or (d3 < -1e-9)
    has_pos = (d1 > 1e-9) or (d2 > 1e-9) or (d3 > 1e-9)
    
    return not (has_neg and has_pos)

def _project_to_triangle(p, tri):
    """Project point p onto the triangle tri."""
    h = np.sqrt(3) / 2
    tri_arr = np.array(tri)
    
    # Use barycentric coordinates
    a, b, c = tri_arr
    
    v0 = b - a
    v1 = c - a
    v2 = p - a
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    
    denom = d00 * d11 - d01 * d01
    if denom < 1e-12:
        return a.copy()
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    
    # If P is inside triangle ABC, we are done
    if v >= -1e-9 and w >= -1e-9 and v + w <= 1 + 1e-9:
        return a + v * v0 + w * v1
    
    # Check if P is outside the triangle, and clip to closest edge
    # Edge AB
    va = d20
    vb = d20 - d00
    if va <= 0:
        if vb >= 0:
            return a
        else:
            return b
    # Edge AC
    va = d21
    vb = d21 - d11
    if va <= 0:
        if vb >= 0:
            return a
        else:
            return c
    # Edge BC
    vc = d00 * d21 - d01 * d20
    vb = d11 * d20 - d01 * d21
    if vb <= 0:
        if vc >= 0:
            return b
        else:
            return c
    
    # Clip to closest edge
    edge = vc - vb
    if edge <= 0:
        alpha = vb / edge
        return b + alpha * (c - b)
    else:
        alpha = vc / edge
        return c + alpha * (b - c)

def _sdp_relaxation(points_init, max_time=60):
    """
    Solve SDP relaxation: maximize t such that ||xi - xj||^2 >= t for all i,j,
    with points constrained to the triangle.
    
    This is a convex surrogate since min triangle area >= (1/4) * min pairwise distance^2
    for equilateral triangles (by geometric inequality).
    """
    start_time = time.time()
    n = N
    h = np.sqrt(3) / 2
    tri = [(0, 0), (1, 0), (0.5, h)]
    tri_arr = np.array(tri)
    
    # We'll use a penalty-based approach with the SDP-inspired objective
    # Maximize minimum pairwise distance squared, with triangle constraint
    
    def _objective_sdp(params):
        points = params.reshape((n, 2))
        
        # Penalty for points outside triangle
        penalty = 0.0
        for p in points:
            if not _is_inside_triangle(p, tri):
                # Distance to triangle boundary (approximate)
                proj = _project_to_triangle(p, tri)
                dist = np.linalg.norm(p - proj)
                penalty += dist * 100.0
        
        if penalty > 1e-6:
            return penalty
        
        # Compute minimum pairwise distance squared
        min_dist_sq = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                d_sq = np.sum((points[i] - points[j]) ** 2)
                if d_sq < min_dist_sq:
                    min_dist_sq = d_sq
        
        return -min_dist_sq
    
    # Multiple random restarts with SDP-inspired initialization
    best_params = None
    best_score = -float('inf')
    
    np.random.seed(123)
    
    for trial in range(30):
        # Create initial points using a grid-like pattern perturbed
        points = np.zeros((n, 2))
        
        # Use a triangular lattice initialization
        for i in range(n):
            # Place points in a triangular grid pattern
            row = i // 4
            col = i % 4
            x = 0.1 + col * 0.2 + np.random.normal(0, 0.02)
            y = 0.1 + row * 0.25 * h + np.random.normal(0, 0.02)
            points[i] = [x, y]
        
        # Ensure points are inside triangle
        for i in range(n):
            if not _is_inside_triangle(points[i], tri):
                points[i] = _project_to_triangle(points[i], tri)
                # Add small inward perturbation
                centroid = np.mean(tri_arr, axis=0)
                points[i] = 0.9 * points[i] + 0.1 * centroid
        
        # Optimize using L-BFGS-B with bounds
        # Define bounds for each coordinate
        bounds = []
        for i in range(n):
            # x range: 0 to 1
            # y range: 0 to h, but constrained by triangle
            bounds.append((0, 1))
            bounds.append((0, h))
        
        try:
            result = minimize(
                _objective_sdp,
                points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            opt_points = result.x.reshape((n, 2))
            
            # Project any out-of-bounds points back to triangle
            for i in range(n):
                if not _is_inside_triangle(opt_points[i], tri):
                    opt_points[i] = _project_to_triangle(opt_points[i], tri)
                    centroid = np.mean(tri_arr, axis=0)
                    opt_points[i] = 0.95 * opt_points[i] + 0.05 * centroid
            
            # Check validity
            valid = all(_is_inside_triangle(p, tri) for p in opt_points)
            if valid:
                score = _min_triangle_area(opt_points)
                if score > best_score:
                    best_score = score
                    best_params = opt_points.copy()
                    
        except Exception:
            continue
        
        if time.time() - start_time > max_time:
            break
    
    return best_params, best_score

def _refine_with_nm(points, max_iter=2000):
    """Refine points using Nelder-Mead on the actual min triangle area objective."""
    n = N
    h = np.sqrt(3) / 2
    tri = [(0, 0), (1, 0), (0.5, h)]
    
    def _objective_nm(params):
        pts = params.reshape((n, 2))
        penalty = 0.0
        for p in pts:
            if not _is_inside_triangle(p, tri):
                proj = _project_to_triangle(p, tri)
                penalty += np.linalg.norm(p - proj) * 10.0
        if penalty > 1e-8:
            return penalty
        return -_min_triangle_area(pts)
    
    try:
        result = minimize(
            _objective_nm,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        opt_points = result.x.reshape((n, 2))
        
        # Project to triangle if needed
        for i in range(n):
            if not _is_inside_triangle(opt_points[i], tri):
                opt_points[i] = _project_to_triangle(opt_points[i], tri)
        
        valid = all(_is_inside_triangle(p, tri) for p in opt_points)
        if valid:
            return opt_points
    except Exception:
        pass
    
    return points

def construct():
    start_time = time.time()
    
    # Equilateral triangle vertices
    h = np.sqrt(3) / 2
    tri = [(0, 0), (1, 0), (0.5, h)]
    tri_arr = np.array(tri)
    
    # Phase 1: SDP relaxation to get a good initial configuration
    sdp_points, sdp_score = _sdp_relaxation(None, max_time=60)
    
    if sdp_points is not None:
        best_points = sdp_points
        best_score = sdp_score
    else:
        # Fallback initialization
        best_points = np.array([
            [0.5, 0.05],
            [0.25, 0.15],
            [0.75, 0.15],
            [0.5, 0.25],
            [0.15, 0.35],
            [0.85, 0.35],
            [0.35, 0.45],
            [0.65, 0.45],
            [0.5, 0.55],
            [0.25, 0.5],
            [0.75, 0.5]
        ])
        best_score = _min_triangle_area(best_points)
    
    # Phase 2: Refine with Nelder-Mead on the true objective
    if time.time() - start_time < 100:
        refined = _refine_with_nm(best_points, max_iter=3000)
        refined_score = _min_triangle_area(refined)
        if refined_score > best_score:
            best_points = refined
            best_score = refined_score
    
    # Phase 3: Additional refinement with multiple small perturbations
    if time.time() - start_time < 110:
        np.random.seed(456)
        for _ in range(10):
            perturbed = best_points + np.random.normal(0, 0.005, (N, 2))
            refined2 = _refine_with_nm(perturbed, max_iter=1000)
            score2 = _min_triangle_area(refined2)
            if score2 > best_score:
                best_points = refined2
                best_score = score2
    
    return best_points
# EVOLVE-BLOCK-END
