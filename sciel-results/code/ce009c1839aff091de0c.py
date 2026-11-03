import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])

def project_to_triangle(p):
    """
    Project a point p (x, y) to the closest point inside the triangle ABC.
    """
    x, y = p
    
    # Helper to project onto a line segment
    def proj_segment(p, a, b):
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        return a + t * ab

    # Check edges
    # Edge AB: y=0
    if y < 0:
        return proj_segment(p, A, B)
        
    # Edge AC: y = sqrt(3)x
    if y > np.sqrt(3) * x:
        return proj_segment(p, A, C)
        
    # Edge BC: y = sqrt(3) * (1 - x)
    if y > np.sqrt(3) * (1 - x):
        return proj_segment(p, B, C)
        
    # If inside, return p
    return p

def compute_min_area(points):
    """
    points: (N, 2) array
    returns: min area of any triangle formed by 3 points
    """
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            v = points[j] - points[i]
            w = points[j+1:] - points[i]
            if len(w) == 0:
                continue
            areas = 0.5 * np.abs(v[0] * w[:, 1] - v[1] * w[:, 0])
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def objective(flat_points):
    """
    Objective function for optimizer.
    Minimizes -min_area.
    """
    points = flat_points.reshape(N, 2)
    # Project points to triangle
    points = np.array([project_to_triangle(p) for p in points])
    
    val = compute_min_area(points)
    # Add a penalty if points are too close to prevent degeneracy
    min_dist = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_dist:
                min_dist = d
    if min_dist < 1e-4:
        return 1.0 # Large positive value to be minimized (bad)
        
    return -val

def construct():
    start_time = time.time()
    time_limit = 120  # seconds, keep margin
    deadline = start_time + time_limit - 5
    
    best_score = 0.0
    best_points = None
    
    def random_init():
        # Random points in the triangle
        pts = np.zeros((N, 2))
        for i in range(N):
            while True:
                x = random.random()
                y = random.random() * np.sqrt(3) / 2
                if y <= np.sqrt(3) * x and y <= np.sqrt(3) * (1 - x):
                    pts[i] = [x, y]
                    break
        return pts

    def grid_init():
        # Create a grid of points and select N
        xs = np.linspace(0.1, 0.9, 4)
        ys = np.linspace(0.1, np.sqrt(3)/2 - 0.1, 3)
        pts = []
        for x in xs:
            for y in ys:
                p = project_to_triangle([x, y])
                pts.append(p)
        idx = random.sample(range(len(pts)), N)
        return np.array([pts[i] for i in idx])

    inits = []
    for _ in range(5):
        inits.append(random_init())
    for _ in range(5):
        inits.append(grid_init())
    
    num_restarts = 0
    max_restarts = 100
    
    while time.time() < deadline:
        if num_restarts < len(inits):
            x0 = inits[num_restarts]
        else:
            if num_restarts % 2 == 0:
                x0 = random_init()
            else:
                x0 = grid_init()
        
        x0_flat = x0.flatten()
        
        try:
            res = minimize(objective, x0_flat, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
            
            pts = res.x.reshape(N, 2)
            pts = np.array([project_to_triangle(p) for p in pts])
            
            score = compute_min_area(pts)
            min_dist = float('inf')
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < min_dist:
                        min_dist = d
            
            if min_dist > 1e-4 and score > best_score:
                best_score = score
                best_points = pts
                
        except Exception as e:
            pass
            
        num_restarts += 1
        
        if time.time() > deadline - 10:
            break

    if best_points is None:
        best_points = grid_init()
        score = compute_min_area(best_points)
        if score < 1e-6:
             best_points = random_init()
        
    return best_points
