import numpy as np
import time

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])

def point_in_triangle(p, v0, v1, v2):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d0 = cross(v0, v1, p)
    d1 = cross(v1, v2, p)
    d2 = cross(v2, v0, p)
    
    has_neg = (d0 < 0) or (d1 < 0) or (d2 < 0)
    has_pos = (d0 > 0) or (d1 > 0) or (d2 > 0)
    
    return not (has_neg and has_pos)

def closest_point_on_segment(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / np.dot(ab, ab)
    t = np.clip(t, 0.0, 1.0)
    return a + t * ab

def project_to_triangle(p):
    if point_in_triangle(p, V0, V1, V2):
        return p
    
    p0 = closest_point_on_segment(p, V0, V1)
    p1 = closest_point_on_segment(p, V1, V2)
    p2 = closest_point_on_segment(p, V2, V0)
    
    d0 = np.linalg.norm(p - p0)
    d1 = np.linalg.norm(p - p1)
    d2 = np.linalg.norm(p - p2)
    
    min_d = min(d0, d1, d2)
    if min_d == d0:
        return p0
    elif min_d == d1:
        return p1
    else:
        return p2

def compute_areas_and_grads(points):
    n = len(points)
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
                
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    A = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(A)
    
    delta = 1e-6
    denom = (areas + delta) ** 2
    signs = np.sign(A)
    coeff = -0.5 * signs / denom
    
    grads = np.zeros_like(points)
    
    gi_x = coeff * (-v2[:, 1])
    gi_y = coeff * (v2[:, 0])
    
    gj_x = coeff * (v2[:, 1])
    gj_y = coeff * (-v2[:, 0])
    
    gk_x = coeff * (-v1[:, 1])
    gk_y = coeff * (v1[:, 0])
    
    np.add.at(grads, (i_idx, 0), gi_x)
    np.add.at(grads, (i_idx, 1), gi_y)
    np.add.at(grads, (j_idx, 0), gj_x)
    np.add.at(grads, (j_idx, 1), gj_y)
    np.add.at(grads, (k_idx, 0), gk_x)
    np.add.at(grads, (k_idx, 1), gk_y)
    
    min_area = np.min(areas)
    
    return min_area, grads

def optimize_candidate(points, steps=100, lr=0.05):
    current_points = points.copy()
    best_points = current_points.copy()
    best_min_area = -1.0
    
    for step in range(steps):
        min_area, grads = compute_areas_and_grads(current_points)
        
        if min_area > best_min_area:
            best_min_area = min_area
            best_points = current_points.copy()
            
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1e-8:
            grads = grads / grad_norm
            
        current_points = current_points - lr * grads
        
        for i in range(N):
            current_points[i] = project_to_triangle(current_points[i])
            
        lr *= 0.995
        
    min_area, _ = compute_areas_and_grads(best_points)
    if min_area > best_min_area:
         best_min_area = min_area
         
    return best_points, best_min_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    rng = np.random.RandomState(42)
    candidates = []
    
    # Generate 30 random candidates
    for _ in range(30):
        pts = []
        while len(pts) < N:
            p = rng.rand(2)
            if point_in_triangle(p, V0, V1, V2):
                pts.append(p)
        candidates.append(np.array(pts))
        
    # Add one structured candidate (Triangular grid n=4 + 1)
    # Barycentric to Cartesian: P = u*V0 + v*V1 + w*V2
    # V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
    # P = (v + 0.5w, (sqrt(3)/2)w)
    
    grid_pts = []
    # (u, v, w) for i+j+k=3
    coords = [
        (1, 0, 0),
        (2/3, 1/3, 0), (2/3, 0, 1/3),
        (1/3, 2/3, 0), (1/3, 1/3, 1/3), (1/3, 0, 2/3),
        (0, 1, 0), (0, 2/3, 1/3), (0, 1/3, 2/3),
        (0, 0, 1)
    ]
    for u, v, w in coords:
        x = v + 0.5 * w
        y = (np.sqrt(3) / 2) * w
        grid_pts.append([x, y])
        
    # Add one point in the center
    grid_pts.append([0.5, np.sqrt(3) / 6]) # Centroid
    
    candidates.append(np.array(grid_pts))
    
    best_points = candidates[0]
    best_score = -1.0
    
    for cand in candidates:
        if time.time() > deadline:
            break
            
        opt_points, opt_score = optimize_candidate(cand, steps=50, lr=0.05)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points
            
    return best_points
