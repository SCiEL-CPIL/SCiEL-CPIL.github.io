import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 14

def _triangle_area(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _convex_hull_area(points):
    # Simple convex hull area calculation using cross products
    # Points must be in convex position or we compute the hull.
    # For speed, we can use a simple O(N^2) or O(N log N) hull.
    # Since N=14, even O(N^3) is fine, but let's do a simple one.
    n = len(points)
    if n < 3:
        return 0.0
    
    # Find the point with lowest y
    start = 0
    for i in range(1, n):
        if points[i][1] < points[start][1] or (points[i][1] == points[start][1] and points[i][0] < points[start][0]):
            start = i
            
    # Sort by polar angle
    def angle_cmp(p):
        return np.arctan2(p[1] - points[start][1], p[0] - points[start][0])
        
    sorted_pts = [points[start]]
    rest = [points[i] for i in range(n) if i != start]
    rest.sort(key=angle_cmp)
    sorted_pts.extend(rest)
    
    # Graham scan
    stack = []
    for p in sorted_pts:
        while len(stack) >= 2:
            p1 = stack[-2]
            p2 = stack[-1]
            cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p[2] if False else p[1]) * 0 # placeholder
            # Correct cross product for 2D
            cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p[0] - p1[0]) * (p2[1] - p1[1])
            if cross <= 1e-10: # Pop if collinear or clockwise (assuming CCW hull)
                stack.pop()
            else:
                break
        stack.append(p)
        
    # Calculate area of polygon defined by stack
    area = 0.0
    m = len(stack)
    for i in range(m):
        x1, y1 = stack[i]
        x2, y2 = stack[(i + 1) % m]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def _min_triangle_area(points):
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def _score(points):
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _geometric_mean_triangle_area(points):
    # For optimization, we maximize the geometric mean of triangle areas
    # This is smooth and differentiable
    log_sum = 0.0
    count = 0
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < 1e-9:
                    return 0.0
                log_sum += np.log(a)
                count += 1
    if count == 0:
        return 0.0
    return np.exp(log_sum / count)

def _objective_geometric(points_flat):
    points = points_flat.reshape(-1, 2)
    gm = _geometric_mean_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 1e9
    return - (gm / hull_area)

def _objective_min(points_flat):
    points = points_flat.reshape(-1, 2)
    score = _score(points)
    return -score

def _grid_init(cols, rows, remove=None):
    pts = []
    for r in range(rows):
        for c in range(cols):
            if remove and (c, r) in remove:
                continue
            pts.append([c / (cols - 1), r / (rows - 1)])
    return np.array(pts)

def _random_init(rng):
    return np.array([[rng.random(), rng.random()] for _ in range(N)])

def _circle_init(rng):
    # Points on a circle
    angles = np.linspace(0, 2*np.pi, N, endpoint=False) + rng.uniform(0, 2*np.pi)
    radii = np.ones(N)
    # Add some noise
    radii += rng.uniform(-0.1, 0.1, N)
    pts = np.column_stack([np.cos(angles) * radii, np.sin(angles) * radii])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Grid 4x4 with 2 removed
    # 14 points in 4x4. Remove (0,0) and (3,3)? Or center ones?
    # Let's try removing corners to maximize spread
    candidates.append(_grid_init(4, 4, remove={(0,0), (3,3)}))
    candidates.append(_grid_init(4, 4, remove={(0,3), (3,0)}))
    candidates.append(_grid_init(4, 4, remove={(1,1), (2,2)}))
    
    # 2. Grid 3x5 with 1 removed
    candidates.append(_grid_init(3, 5, remove={(1,2)})) # Remove center
    candidates.append(_grid_init(5, 3, remove={(2,1)}))
    
    # 3. Random
    for _ in range(10):
        candidates.append(_random_init(rng))
        
    # 4. Circle
    for _ in range(5):
        candidates.append(_circle_init(rng))
        
    # 5. Jittered Grid
    base_grid = _grid_init(4, 4, remove={(0,0), (3,3)})
    for _ in range(5):
        jitter = rng.uniform(-0.1, 0.1, (N, 2))
        candidates.append(base_grid + jitter)

    # Optimize each candidate
    for i, init_pts in enumerate(candidates):
        if time.time() > deadline:
            break
            
        # Normalize initial points to [0,1]x[0,1] roughly
        min_x, min_y = init_pts.min(axis=0)
        max_x, max_y = init_pts.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        if width < 1e-6 or height < 1e-6:
            continue
        init_pts = (init_pts - min_x) / width
        init_pts = (init_pts - min_y) / height # This distorts aspect ratio, but score is affine invariant? No, score is ratio of areas, which is affine invariant. So normalization is safe.
        
        # Actually, affine invariance means we can fix the hull to be a unit square.
        # But let's just optimize in the plane.
        
        x0 = init_pts.flatten()
        
        # Try L-BFGS-B with geometric mean objective
        try:
            res = minimize(_objective_geometric, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success:
                pts = res.x.reshape(-1, 2)
                # Polish with Nelder-Mead on min-area objective
                # Nelder-Mead is slow, so only if time permits
                if time.time() < deadline - 5.0:
                    res2 = minimize(_objective_min, pts.flatten(), method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
                    if res2.success:
                        pts = res2.x.reshape(-1, 2)
                        
                score = _score(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts
        except Exception:
            pass
            
    if best_points is None:
        # Fallback to random
        best_points = _random_init(rng)
        
    return best_points

# EVOLVE-BLOCK-END
