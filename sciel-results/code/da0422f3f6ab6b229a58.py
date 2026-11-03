import numpy as np
import scipy.optimize as so
import time
import itertools

N = 11

def min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points: (N, 2) array
    """
    min_area = float('inf')
    # Iterate over all combinations of 3 points
    for i, j, k in itertools.combinations(range(len(points)), 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        # Area = 0.5 * |det(p2-p1, p3-p1)|
        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        dx2 = p3[0] - p1[0]
        dy2 = p3[1] - p1[1]
        area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
        if area < min_area:
            min_area = area
    return min_area

def points_in_triangle_penalty(points, A, B, C):
    """
    Calculate a penalty if any point is outside the triangle ABC.
    """
    # Barycentric coordinates
    # P = u*A + v*B + w*C, u+v+w=1, u,v,w >= 0
    # Vector form: P = A + v*(B-A) + w*(C-A)
    # Solve for v, w
    AB = B - A
    AC = C - A
    # AP = P - A
    # AP = v*AB + w*AC
    # [AB, AC] [v, w]^T = AP
    # det = ABx*ACy - ABy*ACx
    det = AB[0]*AC[1] - AB[1]*AC[0]
    if abs(det) < 1e-10:
        return 0.0
        
    penalty = 0.0
    for P in points:
        AP = P - A
        v = (AP[0]*AC[1] - AP[1]*AC[0]) / det
        w = (AB[0]*AP[1] - AB[1]*AP[0]) / det
        u = 1 - v - w
        if u < -1e-6 or v < -1e-6 or w < -1e-6:
            # Penalty proportional to how far out
            dist_out = max(0, -u, -v, -w)
            penalty += dist_out ** 2 * 100.0
    return penalty

def construct():
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_points = None
    best_score = -float('inf')
    
    def objective(x):
        points = x.reshape(-1, 2)
        area = min_triangle_area(points)
        pen = points_in_triangle_penalty(points, A, B, C)
        return -area + pen
        
    # Initial configurations
    inits = []
    
    # 1. Grid-like
    # 11 points. 3x3 grid + 2?
    # Let's try a hexagonal packing approximation
    # Or just random with a fixed seed
    rng = np.random.RandomState(42)
    
    # Config 1: Random
    p1 = rng.rand(N, 2)
    # Map to triangle: u,v > 0, u+v < 1
    # P = u*A + v*B + (1-u-v)*C ? No, barycentric weights must be non-negative.
    # Let's use: u, v random in [0,1]. If u+v > 1, reject.
    # This is rejection sampling.
    pts = []
    while len(pts) < N:
        u, v = rng.rand(2)
        if u + v < 1:
            pts.append(A + u*(B-A) + v*(C-A))
    inits.append(np.array(pts))
    
    # Config 2: Structured
    # 3 rows: 3, 4, 4? No, 11 points.
    # Maybe 2, 4, 5?
    # Let's just use a few random restarts with different seeds
    for seed in [1, 2, 3, 4, 5]:
        rng = np.random.RandomState(seed)
        pts = []
        while len(pts) < N:
            u, v = rng.rand(2)
            if u + v < 1:
                pts.append(A + u*(B-A) + v*(C-A))
        inits.append(np.array(pts))
        
    # Also try a "centered" random
    rng = np.random.RandomState(99)
    pts = []
    while len(pts) < N:
        u, v = rng.rand(2)
        if u + v < 1:
            # Bias towards center?
            pts.append(A + u*(B-A) + v*(C-A))
    inits.append(np.array(pts))

    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = so.minimize(objective, init.flatten(), method='Nelder-Mead', 
                              options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False})
            
            if time.time() > deadline:
                break
                
            points = res.x.reshape(-1, 2)
            # Check validity
            if points_in_triangle_penalty(points, A, B, C) < 1e-5:
                score = min_triangle_area(points)
                if score > best_score:
                    best_score = score
                    best_points = points
        except Exception:
            pass
            
    if best_points is None:
        # Fallback to seed-like random
        return inits[0]
        
    return best_points
