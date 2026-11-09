"""Rewritten program for Heilbronn triangles task using lattice search + local polish."""
import numpy as np
from scipy.optimize import minimize
import itertools

N = 11

# EVOLVE-BLOCK-START
def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three 2D points."""
    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))


def _min_triangle_area(points):
    """Compute the minimum area over all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area


def _inside_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the reference triangle."""
    # Reference triangle: (0,0), (1,0), (0.5, sqrt(3)/2)
    x, y = p
    h = np.sqrt(3) / 2
    # Barycentric coordinates
    # A = (0,0), B = (1,0), C = (0.5, h)
    # Area of ABC
    area_abc = 0.5 * h
    # Area of PBC
    area_pbc = 0.5 * abs((1 - x) * h - (0 - y) * (0.5 - x))
    # More robust: use cross products
    # For triangle with vertices A, B, C, point P is inside if it's on the same side of each edge
    # Edge AB: from (0,0) to (1,0). C is above. P must be above or on AB.
    # y >= 0
    if y < -eps:
        return False
    # Edge BC: from (1,0) to (0.5, h). A is to the left.
    # Vector BC = (-0.5, h), BA = (-1, 0)
    # Cross(BC, BP) where BP = (x-1, y)
    # cross = (-0.5)*y - h*(x-1) = -0.5*y - h*x + h
    # A should give: -0.5*0 - h*0 + h = h > 0
    # So we need -0.5*y - h*x + h >= 0
    if -0.5 * y - h * x + h < -eps:
        return False
    # Edge CA: from (0.5, h) to (0,0). B is to the right.
    # Vector CA = (-0.5, -h), CB = (0.5, -h)
    # Cross(CA, CP) where CP = (x-0.5, y-h)
    # cross = (-0.5)*(y-h) - (-h)*(x-0.5) = -0.5*y + 0.5*h + h*x - 0.5*h = -0.5*y + h*x
    # B should give: -0.5*0 + h*1 = h > 0
    # So we need -0.5*y + h*x >= 0
    if -0.5 * y + h * x < -eps:
        return False
    return True


def _project_to_triangle(p):
    """Project point p onto the reference triangle (clip to feasible region)."""
    x, y = p
    h = np.sqrt(3) / 2
    
    # Simple approach: if outside, find closest point on boundary
    # For simplicity, we'll just check and adjust
    if not _inside_triangle(p):
        # Find the closest point on the triangle boundary
        # This is complex; for now, just scale toward centroid
        cx, cy = 0.5, h / 3
        # Move toward centroid until inside
        t = 0.0
        while t < 1.0:
            t += 0.01
            qx = x + t * (cx - x)
            qy = y + t * (cy - y)
            if _inside_triangle([qx, qy]):
                return np.array([qx, qy])
        return np.array([cx, cy])
    return np.array([x, y])


def construct():
    """
    Construct 11 points inside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)
    to maximize the smallest triangle area.
    """
    h = np.sqrt(3) / 2
    centroid = np.array([0.5, h / 3])
    
    # Step 1: Generate initial candidate set using barycentric grid
    # Use a grid of barycentric coordinates (u, v, w) with u+v+w=1, u,v,w >= 0
    # Map to Cartesian: P = u*A + v*B + w*C where A=(0,0), B=(1,0), C=(0.5,h)
    
    # Create a set of candidate points from a moderate grid
    grid_n = 12  # number of divisions
    candidates = []
    for i in range(grid_n + 1):
        for j in range(grid_n + 1 - i):
            k = grid_n - i - j
            u, v, w = i / grid_n, j / grid_n, k / grid_n
            x = 0 * u + 1 * v + 0.5 * w
            y = 0 * u + 0 * v + h * w
            candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    # Step 2: Greedy selection to maximize minimum pairwise distance (packing)
    # This gives a good initial configuration
    selected = []
    # Start with centroid
    selected.append(centroid.copy())
    
    for _ in range(N - 1):
        best_candidate = None
        best_min_dist = -1
        for cand in candidates:
            if len(selected) > 0:
                dists = [np.linalg.norm(cand - s) for s in selected]
                min_dist = min(dists)
            else:
                min_dist = np.linalg.norm(cand - centroid)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = cand.copy()
        selected.append(best_candidate)
    
    points = np.array(selected)
    
    # Step 3: Local optimization to maximize minimum triangle area
    # Flatten points for optimization
    def _neg_min_triangle_area(flat_points):
        pts = flat_points.reshape(-1, 2)
        # Project each point to ensure feasibility
        pts = np.array([_project_to_triangle(p) for p in pts])
        return -_min_triangle_area(pts)
    
    x0 = points.flatten()
    
    # Use Nelder-Mead for the non-smooth objective
    best_result = None
    best_score = -float('inf')
    
    # Run multiple restarts from slightly perturbed initial points
    rng = np.random.RandomState(42)
    
    for restart in range(5):
        if restart == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + rng.normal(0, 0.02, size=x0.shape)
            # Project to triangle
            x_start = np.array([_project_to_triangle(p).flatten() for p in x_start.reshape(-1, 2)]).flatten()
        
        result = minimize(
            _neg_min_triangle_area,
            x_start,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        
        score = -result.fun
        if score > best_score:
            best_score = score
            best_result = result.x
    
    final_points = np.array([_project_to_triangle(p) for p in best_result.reshape(-1, 2)])
    
    return final_points
# EVOLVE-BLOCK-END
