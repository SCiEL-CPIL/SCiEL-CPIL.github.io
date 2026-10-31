import numpy as np
from scipy.optimize import minimize
import time

N = 11
S3 = np.sqrt(3)
H = S3 / 2.0

def triangle_area(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    return 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])

def compute_min_area(points):
    areas = []
    for i in range(N):
        pi = points[i]
        for j in range(i+1, N):
            pj = points[j]
            v1 = pj - pi
            for k in range(j+1, N):
                pk = points[k]
                v2 = pk - pi
                areas.append(0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0]))
    return np.array(areas)

def objective(params, beta, penalty_weight):
    points = params.reshape(-1, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
        
    areas = compute_min_area(points)
    min_area_val = np.min(areas)
    if min_area_val < 1e-9:
        return 1e10
    
    inv_areas = 1.0 / areas
    sum_inv_beta = np.sum(inv_areas ** beta)
    soft_min = (sum_inv_beta) ** (-1.0/beta)
    
    x = points[:, 0]
    y = points[:, 1]
    
    d1 = y
    d2 = S3 * x - y
    d3 = S3 * (1.0 - x) - y
    
    viol1 = np.maximum(0, -d1)
    viol2 = np.maximum(0, -d2)
    viol3 = np.maximum(0, -d3)
    
    penalty = np.sum(viol1**2) + np.sum(viol2**2) + np.sum(viol3**2)
    
    return -soft_min + penalty_weight * penalty

def random_point_in_triangle():
    u = np.random.random()
    v = np.random.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    x = v * 1.0 + (1-u-v) * 0.5
    y = (1-u-v) * H
    return [x, y]

def hex_lattice_init(seed=0):
    np.random.seed(seed)
    pts = []
    a = 0.35
    for j in range(0, 6):
        for i in range(0, 6):
            x = i * a + (j % 2) * a/2
            y = j * a * H
            if y >= -1e-6 and y <= S3 * x + 1e-6 and y <= S3 * (1 - x) + 1e-6:
                pts.append([x, y])
                
    if len(pts) > N:
        idxs = np.random.choice(len(pts), N, replace=False)
        pts = [pts[i] for i in idxs]
    elif len(pts) < N:
        while len(pts) < N:
            pts.append(random_point_in_triangle())
            
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_params = None
    best_score = -np.inf
    
    betas = [2, 4, 8, 16, 32, 64]
    
    # Try a few initializations
    inits = [
        hex_lattice_init(0),
        hex_lattice_init(1),
        np.array([random_point_in_triangle() for _ in range(N)]),
        np.array([random_point_in_triangle() for _ in range(N)])
    ]
    
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        current_params = init.flatten()
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
                
            res = minimize(objective, current_params, args=(beta, 10.0), 
                           method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success:
                current_params = res.x
                points = current_params.reshape(-1, 2)
                areas = compute_min_area(points)
                score = np.min(areas)
                if score > best_score:
                    best_score = score
                    best_params = current_params.copy()
            else:
                # If it fails, maybe try a smaller step or break
                pass
                
    if best_params is None:
        best_params = inits[0].flatten()
        
    return best_params.reshape(-1, 2)
