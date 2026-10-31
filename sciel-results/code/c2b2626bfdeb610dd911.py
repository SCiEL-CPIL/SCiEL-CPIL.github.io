"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Triangle vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    # Helper: Calculate area of triangle formed by 3 points
    def area(p1, p2, p3):
        return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    
    # Helper: Calculate minimum area among all triples
    def min_area(points):
        min_a = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = area(points[i], points[j], points[k])
                    if a < min_a:
                        min_a = a
        return min_a

    # Helper: Check if a point is inside the triangle
    def is_inside(p):
        x, y = p
        if y < 0: return False
        if y > np.sqrt(3)*x: return False
        if y > np.sqrt(3)*(1-x): return False
        return True

    # Helper: Project a point onto the triangle (simple version: if outside, move towards centroid)
    def project_point(p):
        if is_inside(p):
            return p
        # Simple projection: move towards centroid until inside
        centroid = (A + B + C) / 3.0
        for t in np.linspace(0, 1, 100):
            p_test = p * (1 - t) + centroid * t
            if is_inside(p_test):
                return p_test
        return centroid

    # Helper: Penalty for points outside the triangle (stronger version)
    def penalty(points):
        pen = 0.0
        for p in points:
            x, y = p
            # y >= 0
            if y < 0: pen += 100 * y*y
            # y <= sqrt(3)*x
            if y > np.sqrt(3)*x: pen += 100 * (y - np.sqrt(3)*x)**2
            # y <= sqrt(3)*(1-x)
            if y > np.sqrt(3)*(1-x): pen += 100 * (y - np.sqrt(3)*(1-x))**2
        return pen

    # Objective function: Minimize sum of inverse squared areas (smooth energy) + penalty
    # This transforms the max-min problem into a smooth minimization problem
    def objective(params):
        points = params.reshape((N, 2))
        
        # Calculate all triangle areas
        areas = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = area(points[i], points[j], points[k])
                    # Avoid division by zero
                    if a < 1e-9:
                        a = 1e-9
                    areas.append(1.0 / (a * a))
        
        energy = np.sum(areas)
        p_pen = penalty(points)
        
        return energy + p_pen

    # Generate initial guesses
    inits = []
    rng = np.random.RandomState(42)
    
    # 1. Random points
    for _ in range(10):
        pts = []
        while len(pts) < N:
            u, v = rng.random(2)
            if u + v < 1:
                p = u*A + v*B + (1-u-v)*C
                pts.append(p)
        inits.append(np.array(pts).flatten())
        
    # 2. Grid-like initialization
    grid_pts = []
    for i in range(4):
        for j in range(4-i):
            u = i/3.0
            v = j/3.0
            w = 1 - u - v
            if u >= 0 and v >= 0 and w >= 0:
                p = u*A + v*B + w*C
                grid_pts.append(p)
    center = (A+B+C)/3.0
    grid_pts.append(center)
    inits.append(np.array(grid_pts).flatten())
    
    # 3. Symmetric initialization
    center = (A+B+C)/3.0
    pts = [center]
    angle_step = 2 * np.pi / 10
    radius = 0.25
    for i in range(10):
        angle = i * angle_step
        p = center + radius * np.array([np.cos(angle), np.sin(angle)])
        if not is_inside(p):
            p = project_point(p)
        pts.append(p)
    inits.append(np.array(pts).flatten())

    best_score = -float('inf')
    best_points = None

    for init in inits:
        if time.time() > deadline:
            break
            
        # Use Powell optimizer on the smooth energy function
        res = minimize(objective, init, method='Powell', options={'maxiter': 100, 'xtol': 1e-6, 'ftol': 1e-6})
        
        points = res.x.reshape((N, 2))
        
        # Project points to ensure they are strictly inside
        valid_points = []
        for p in points:
            valid_points.append(project_point(p))
        valid_points = np.array(valid_points)
        
        # Check validity
        if all(is_inside(p) for p in valid_points):
            score = min_area(valid_points)
            if score > best_score:
                best_score = score
                best_points = valid_points

    if best_points is None:
        # Fallback to first init if optimization failed
        best_points = inits[0].reshape((N, 2))
        # Ensure fallback is valid
        valid_points = []
        for p in best_points:
            valid_points.append(project_point(p))
        best_points = np.array(valid_points)
        
    return best_points
# EVOLVE-BLOCK-END
