"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Triangle vertices: (0,0), (1,0), (1/2, sqrt(3)/2)
H = np.sqrt(3) / 2
TRI_AREA = H  # area = 1/2 * base * height = 1/2 * 1 * H = H/2... wait
# Actually area = 0.5 * 1 * H = H/2 = sqrt(3)/4
# But the problem says "unit-area triangle" and "divides by its area"
# The evaluator measures raw value and divides by seed. Let me re-read.
# "The notebook uses an equilateral triangle and divides by its area."
# So the score is min_triangle_area / triangle_area
# Triangle area = sqrt(3)/4
TRI_AREA = np.sqrt(3) / 4.0


def _min_triangle_area(points):
    """Compute the minimum area among all C(n,3) triangles formed by the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """Score = min_triangle_area / triangle_area (normalized)."""
    return _min_triangle_area(points) / TRI_AREA


def _project_to_triangle(p):
    """Project a point to be inside the triangle (0,0), (1,0), (1/2, H)."""
    # Use barycentric coordinates
    # Triangle: A=(0,0), B=(1,0), C=(0.5, H)
    # For point P, barycentric coords (a,b,c) with a+b+c=1
    # P = a*A + b*B + c*C
    # x = b + 0.5*c, y = H*c
    # So c = y/H, b = x - 0.5*c, a = 1 - b - c
    x, y = p[0], p[1]
    c = y / H
    b = x - 0.5 * c
    a = 1.0 - b - c
    # If any coordinate is negative, project to nearest edge
    if a < 0:
        a = 0.0
        # Renormalize b and c
        s = b + c
        if s > 0:
            b /= s
            c /= s
    if b < 0:
        b = 0.0
        s = a + c
        if s > 0:
            a /= s
            c /= s
    if c < 0:
        c = 0.0
        s = a + b
        if s > 0:
            a /= s
            b /= s
    # Reconstruct point
    px = b * 1.0 + c * 0.5
    py = c * H
    return np.array([px, py])


def _neg_score(x):
    """Negative score for minimization."""
    points = x.reshape(-1, 2)
    return -_score(points)


def _gradient(x, eps=1e-7):
    """Numerical gradient of negative score."""
    points = x.reshape(-1, 2)
    f0 = _neg_score(x)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_pert = x.copy()
        x_pert[i] += eps
        grad[i] = (_neg_score(x_pert) - f0) / eps
    return grad


def _optimize_from_start(start_points, max_time=10.0):
    """Optimize starting from initial points using Nelder-Mead and then BFGS."""
    x0 = start_points.flatten().copy()
    t_start = time.time()
    
    # Phase 1: Nelder-Mead (robust, doesn't need gradients)
    result1 = minimize(_neg_score, x0, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10,
                                'adaptive': True})
    best_x = result1.x
    best_f = result1.fun
    
    if time.time() - t_start > max_time:
        return best_x.reshape(-1, 2)
    
    # Phase 2: Try BFGS with numerical gradient
    try:
        result2 = minimize(_neg_score, best_x, method='BFGS',
                           jac=_gradient,
                           options={'maxiter': 50, 'gtol': 1e-8})
        if result2.fun < best_f:
            best_x = result2.x
            best_f = result2.fun
    except Exception:
        pass
    
    if time.time() - t_start > max_time:
        return best_x.reshape(-1, 2)
    
    # Phase 3: Another round of Nelder-Mead from best
    try:
        result3 = minimize(_neg_score, best_x, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-9, 'fatol': 1e-12,
                                    'adaptive': True})
        if result3.fun < best_f:
            best_x = result3.x
            best_f = result3.fun
    except Exception:
        pass
    
    return best_x.reshape(-1, 2)


def _make_symmetric_config():
    """
    Create a configuration with symmetry. For 11 points in an equilateral triangle,
    a good starting configuration uses the symmetry of the triangle.
    
    Known good configurations for Heilbronn's problem use points arranged with
    3-fold symmetry or mirror symmetry.
    
    For n=11, one approach: place points on a grid-like pattern that respects
    the triangle's symmetry.
    """
    # Start with a hexagonal/triangular lattice pattern
    # and add boundary points
    
    # Use a structured grid inside the triangle
    points = []
    
    # Center point
    cx, cy = 1/3, H/3
    points.append([cx, cy])
    
    # Add points in concentric layers with 6-fold symmetry around center
    # Layer 1: 6 points
    r1 = 0.15
    for angle in [0, 60, 120, 180, 240, 300]:
        rad = np.radians(angle)
        px = cx + r1 * np.cos(rad)
        py = cy + r1 * np.sin(rad)
        points.append([px, py])
    
    # Layer 2: 4 more points (to get to 11 total)
    r2 = 0.30
    for angle in [30, 150, 270, 330]:
        rad = np.radians(angle)
        px = cx + r2 * np.cos(rad)
        py = cy + r2 * np.sin(rad)
        points.append([px, py])
    
    # Project all points to triangle
    points = np.array([_project_to_triangle(p) for p in points])
    return points


def _make_grid_config():
    """Create a grid-based configuration."""
    points = []
    # Use a triangular grid
    # Row 0 (bottom): points along y=0
    # Row 1: points at y = H/4
    # Row 2: points at y = H/2
    # Row 3: points at y = 3H/4
    
    # Let's try a 3-3-3-2 pattern (11 points)
    # Bottom row (y=0.05): 3 points
    for x in [0.1, 0.5, 0.9]:
        points.append([x, 0.05])
    # Second row (y=H/4): 3 points
    for x in [0.2, 0.5, 0.8]:
        points.append([x, H/4])
    # Third row (y=H/2): 3 points
    for x in [0.25, 0.5, 0.75]:
        points.append([x, H/2])
    # Top row (y=3H/4): 2 points
    for x in [0.35, 0.65]:
        points.append([x, 3*H/4])
    
    points = np.array([_project_to_triangle(p) for p in points])
    return points


def _make_random_config(seed):
    """Create a random configuration."""
    rng = np.random.RandomState(seed)
    points = []
    while len(points) < N:
        u, v = rng.random(), rng.random()
        if u + v < 1.0 and u > 0.02 and v > 0.02:
            px = u + 0.5 * v
            py = H * v
            points.append([px, py])
    return np.array(points)


def construct():
    """Place 11 points in the equilateral triangle to maximize min triangle area."""
    t_start = time.time()
    DEADLINE = 110.0  # Leave 10s safety margin
    
    # Try multiple starting configurations
    start_configs = []
    
    # 1. Symmetric hexagonal config
    start_configs.append(_make_symmetric_config())
    
    # 2. Grid config
    start_configs.append(_make_grid_config())
    
    # 3. Several random configs
    for seed in [42, 123, 456, 789, 1000]:
        start_configs.append(_make_random_config(seed))
    
    best_points = None
    best_score = -1.0
    
    for i, config in enumerate(start_configs):
        if time.time() - t_start > DEADLINE:
            break
        
        # Check if config is valid (all points in triangle)
        valid = True
        for p in config:
            if p[1] < -1e-10 or p[0] < -1e-10 or p[0] > 1 + 1e-10:
                valid = False
                break
            # Check if inside triangle: y <= H * (1 - |x - 0.5|) for x in [0,1]
            # Actually: for the triangle, the constraint is y <= H*(1-x) for x>=0.5 
            # and y <= H*x for x<=0.5
            if p[0] <= 0.5:
                if p[1] > H * p[0] + 1e-10:
                    valid = False
                    break
            else:
                if p[1] > H * (1 - p[0]) + 1e-10:
                    valid = False
                    break
        
        if not valid:
            # Project and check again
            config = np.array([_project_to_triangle(p) for p in config])
        
        # Optimize
        try:
            optimized = _optimize_from_start(config, max_time=8.0)
            score = _score(optimized)
            if score > best_score:
                best_score = score
                best_points = optimized
        except Exception:
            continue
    
    # Final check: if we never got a valid result, use the best raw config
    if best_points is None:
        best_points = start_configs[0]
    
    # Ensure all points are strictly inside the triangle
    final_points = np.array([_project_to_triangle(p) for p in best_points])
    
    return final_points


# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
