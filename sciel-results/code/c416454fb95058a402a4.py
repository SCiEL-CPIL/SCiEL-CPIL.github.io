"""SDP relaxation approach for Heilbronn triangles."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools
import time

def _triangle_area(p1, p2, p3):
    """Compute area of triangle given three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = float('inf')
    # Vectorized computation for speed
    # points shape (n, 2)
    # We need all combinations of 3
    # For n=11, C(11,3) = 165, which is small.
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
    
    if v >= -1e-9 and w >= -1e-9 and v + w <= 1 + 1e-9:
        return a + v * v0 + w * v1
    
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
    
    edge = vc - vb
    if edge <= 0:
        alpha = vb / edge
        return b + alpha * (c - b)
    else:
        alpha = vc / edge
        return c + alpha * (b - c)

def _objective_true(params, tri):
    """
    Objective function for optimization.
    Returns negative min triangle area + penalty for being outside triangle.
    """
    points = params.reshape((N, 2))
    
    # Penalty for points outside triangle
    penalty = 0.0
    for p in points:
        if not _is_inside_triangle(p, tri):
            proj = _project_to_triangle(p, tri)
            dist = np.linalg.norm(p - proj)
            penalty += dist * 1000.0  # Large penalty
    
    if penalty > 1e-6:
        return penalty
    
    return -_min_triangle_area(points)

def _local_refine(points, tri, max_iter=2000):
    """Refine points using L-BFGS-B on the true objective."""
    n = N
    h = np.sqrt(3) / 2
    bounds = []
    for i in range(n):
        bounds.append((0, 1))
        bounds.append((0, h))
        
    try:
        result = minimize(
            _objective_true,
            points.flatten(),
            args=(tri,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
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
    
    h = np.sqrt(3) / 2
    tri = [(0, 0), (1, 0), (0.5, h)]
    tri_arr = np.array(tri)
    
    # Global search using Differential Evolution
    # Bounds for each coordinate
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, h))
        
    best_points = None
    best_score = -float('inf')
    
    # Run Differential Evolution
    # We use a penalty-based objective for DE as well, or just rely on bounds
    # DE respects bounds. The penalty handles the triangle constraint.
    
    np.random.seed(42)
    
    try:
        de_result = differential_evolution(
            _objective_true,
            bounds=bounds,
            args=(tri,),
            maxiter=100,
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=True, # Uses L-BFGS-B at the end
            update_interval=1
        )
        
        if de_result.success or de_result.fun < best_score:
            de_points = de_result.x.reshape((N, 2))
            for i in range(N):
                if not _is_inside_triangle(de_points[i], tri):
                    de_points[i] = _project_to_triangle(de_points[i], tri)
            
            valid = all(_is_inside_triangle(p, tri) for p in de_points)
            if valid:
                score = _min_triangle_area(de_points)
                if score > best_score:
                    best_score = score
                    best_points = de_points.copy()
                    
    except Exception as e:
        pass
    
    # If DE failed or didn't find anything, use a grid initialization
    if best_points is None:
        # Grid initialization
        grid_points = []
        # 4x3 grid approx
        for i in range(4):
            for j in range(3):
                x = 0.15 + i * 0.2
                y = 0.1 + j * 0.25 * h
                if _is_inside_triangle([x, y], tri):
                    grid_points.append([x, y])
        # Add more if needed
        while len(grid_points) < N:
            grid_points.append([0.5, h/2])
        best_points = np.array(grid_points[:N])
        best_score = _min_triangle_area(best_points)
        
    # Refine the best point found
    if time.time() - start_time < 110:
        refined = _local_refine(best_points, tri, max_iter=5000)
        refined_score = _min_triangle_area(refined)
        if refined_score > best_score:
            best_points = refined
            best_score = refined_score
            
    # Try a few more random starts if time permits
    if time.time() - start_time < 115:
        for seed in [1, 2, 3]:
            if time.time() - start_time > 115:
                break
            np.random.seed(seed)
            random_points = np.random.uniform(0, 1, (N, 2))
            random_points[:, 1] *= h
            # Project to triangle
            for i in range(N):
                if not _is_inside_triangle(random_points[i], tri):
                    random_points[i] = _project_to_triangle(random_points[i], tri)
            
            refined_rand = _local_refine(random_points, tri, max_iter=2000)
            rand_score = _min_triangle_area(refined_rand)
            if rand_score > best_score:
                best_points = refined_rand
                best_score = rand_score
                
    return best_points
# EVOLVE-BLOCK-END
