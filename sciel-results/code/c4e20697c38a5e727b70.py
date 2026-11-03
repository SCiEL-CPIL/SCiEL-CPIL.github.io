import numpy as np
from scipy.spatial import ConvexHull

N = 14

# EVOLVE-BLOCK-START
def _triangle_area(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    return 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])

def _score(points):
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return -1e6
    except Exception:
        return -1e6
        
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
                    
    if min_area < 1e-9:
        return -1e6
        
    return min_area / hull_area

def _sa_optimize(start_points, iterations=5000, t0=1.0, t_end=1e-4, seed=0):
    rng = np.random.RandomState(seed)
    current_pts = start_points.copy()
    current_cost = -_score(current_pts)
    best_pts = current_pts.copy()
    best_cost = current_cost
    
    if np.isnan(current_cost) or current_cost > 0:
        current_cost = 1e6
        best_cost = 1e6

    for i in range(iterations):
        # Cooling schedule
        t = t0 * (t_end / t0) ** (i / iterations)
        
        # Generate move: perturb 1 or 2 random points
        num_moves = rng.choice([1, 2], p=[0.7, 0.3])
        indices = rng.choice(N, size=num_moves, replace=False)
        
        new_pts = current_pts.copy()
        # Step size decreases with temperature? Or fixed? 
        # Let's use a step size proportional to sqrt(t) to allow large moves early
        step_size = 0.1 * np.sqrt(t) + 0.01
        noise = rng.normal(0, step_size, size=(num_moves, 2))
        new_pts[indices] += noise
        
        new_cost = -_score(new_pts)
        
        if np.isnan(new_cost) or new_cost > 1e6:
            new_cost = 1e6 + 1
            
        # Acceptance criterion
        if new_cost < current_cost:
            current_pts = new_pts
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_pts = current_pts.copy()
        else:
            delta = new_cost - current_cost
            if delta < 0:
                accept = True
            else:
                accept_prob = np.exp(-delta / t)
                accept = (rng.rand() < accept_prob)
                
            if accept:
                current_pts = new_pts
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_pts = current_pts.copy()
                
    return best_pts, -best_cost

def construct():
    best_points = None
    best_score = -1e6
    
    # Strategy 1: Regular Polygon start
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Run multiple SA instances with different seeds and starting points
    for seed in range(10):
        # Perturb the regular polygon slightly to break symmetry
        rng = np.random.RandomState(seed)
        noise = rng.normal(0, 0.05, size=(N, 2))
        start_pts = reg_pts + noise
        
        pts, score = _sa_optimize(start_pts, iterations=3000, t0=1.0, t_end=1e-5, seed=seed)
        if score > best_score:
            best_score = score
            best_points = pts
            
    # Strategy 2: Random start
    for seed in range(5, 15):
        rng = np.random.RandomState(seed)
        start_pts = rng.uniform(-1, 1, size=(N, 2))
        
        pts, score = _sa_optimize(start_pts, iterations=3000, t0=1.0, t_end=1e-5, seed=seed)
        if score > best_score:
            best_score = score
            best_points = pts
            
    if best_points is None:
        return reg_pts
        
    return best_points
# EVOLVE-BLOCK-END
