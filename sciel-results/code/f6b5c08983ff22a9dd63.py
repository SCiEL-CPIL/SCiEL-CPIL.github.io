import numpy as np
import time
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Triangle vertices
    # A = (0, 0)
    # B = (1, 0)
    # C = (0.5, sqrt(3)/2)
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    vertices = np.array([A, B, C])
    
    def bary_to_cart(u, v):
        # u, v are barycentric coordinates for B and C, w = 1 - u - v for A
        # P = u*B + v*C + w*A
        # Since A is origin, P = u*B + v*C
        # Constraints: u >= 0, v >= 0, u+v <= 1
        return u * B + v * C

    def get_points_from_bary(bary_coords):
        # bary_coords shape (N, 2) representing u, v
        # P_i = u_i * B + v_i * C
        return bary_coords[:, 0] * B + bary_coords[:, 1] * C

    def min_triangle_area(points):
        # points shape (N, 2)
        # Calculate area for all triplets
        # To be efficient, we can use broadcasting
        # But N=11, so 11*10*9/6 = 165 triangles. This is small.
        areas = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    p1, p2, p3 = points[i], points[j], points[k]
                    # Area = 0.5 * |det([p2-p1, p3-p1])|
                    v1 = p2 - p1
                    v2 = p3 - p1
                    det = v1[0]*v2[1] - v1[1]*v2[0]
                    areas.append(abs(det) * 0.5)
        return np.min(areas)

    def objective(bary_coords):
        # bary_coords shape (N, 2)
        # Check constraints: u >= 0, v >= 0, u+v <= 1
        # If violated, return a large penalty
        if np.any(bary_coords < -1e-9) or np.any(bary_coords.sum(axis=1) > 1 + 1e-9):
            # Soft penalty to guide back inside
            penalty = 0
            for i in range(N):
                u, v = bary_coords[i]
                if u < 0: penalty += (100 - u)**2
                if v < 0: penalty += (100 - v)**2
                if u + v > 1: penalty += (u + v - 1 + 100)**2
            return 1e6 + penalty
        
        points = get_points_from_bary(bary_coords)
        return -min_triangle_area(points)

    # Initializations
    inits = []
    
    # 1. Random initialization
    rng = np.random.default_rng(42)
    for _ in range(10):
        # Generate random points in unit triangle
        # u, v > 0, u+v < 1
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1:
                break
        # Scale to fill triangle better? 
        # Just use u, v directly.
        # Actually, uniform in barycentric coords is not uniform in area?
        # Let's just generate random Cartesian points and convert?
        # Easier: generate random u, v such that u,v >= 0, u+v <= 1.
        # Uniform in (u,v) triangle is fine.
        
    # Better: Generate N random points inside the triangle
    def random_points_in_triangle(n, seed):
        r = np.random.default_rng(seed)
        pts = []
        for _ in range(n):
            while True:
                u = r.random()
                v = r.random()
                if u + v <= 1:
                    pts.append([u, v])
                    break
        return np.array(pts)
    
    # Add random inits
    for seed in range(20):
        inits.append(random_points_in_triangle(N, seed))
        
    # Add structured inits: Grid-like
    # Place points roughly on a grid in barycentric coords
    # For N=11, maybe 4x3 grid?
    # Let's try a few heuristic placements
    # Center point + ring?
    # Just rely on optimizer with good random starts.
    
    best_val = -np.inf
    best_bary = None
    
    # Run optimization for each init
    for i, init_bary in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Powell is good for non-smooth
        try:
            res = minimize(objective, init_bary, method='Powell', 
                           options={'maxiter': 500, 'disp': False})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_bary = res.x
        except Exception as e:
            pass

    # If we have time, refine the best one with Nelder-Mead
    if best_bary is not None and time.time() < deadline - 10:
        try:
            res2 = minimize(objective, best_bary, method='Nelder-Mead',
                            options={'maxiter': 1000, 'disp': False})
            val2 = -res2.fun
            if val2 > best_val:
                best_val = val2
                best_bary = res2.x
        except:
            pass

    # Convert best_bary to Cartesian points
    if best_bary is None:
        # Fallback: random points
        best_bary = random_points_in_triangle(N, 123)
        
    # Ensure points are strictly inside (clip tiny negatives)
    best_bary = np.clip(best_bary, 0, 1)
    # Normalize if sum > 1? Powell might drift slightly out.
    sums = best_bary.sum(axis=1)
    mask = sums > 1
    if np.any(mask):
        best_bary[mask] /= sums[mask, None]
        
    points = get_points_from_bary(best_bary)
    return points

