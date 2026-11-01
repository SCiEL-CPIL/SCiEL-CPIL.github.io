import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

def compute_radius(points):
    if not np.all(np.isfinite(points)):
        return 0.0, False
    
    dists_center = np.linalg.norm(points, axis=1)
    boundary_r = np.min(1.0 - dists_center)
    
    if boundary_r <= 0:
        return 0.0, False
        
    dist_matrix = cdist(points, points)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair_dist = np.min(dist_matrix)
    pair_r = min_pair_dist / 2.0
    
    if pair_r <= 0:
        return 0.0, False
        
    r = min(boundary_r, pair_r)
    
    if r <= 0 or not np.isfinite(r):
        return 0.0, False
        
    return r, True

def objective(points_flat):
    points = points_flat.reshape(-1, 2)
    r, valid = compute_radius(points)
    if not valid:
        return 1e10
    return -r

def optimize_pack(initial_points, time_limit=5.0):
    start_time = time.time()
    best_points = initial_points.copy()
    best_r, _ = compute_radius(best_points)
    
    try:
        res = minimize(objective, initial_points.flatten(), method='Powell', 
                       options={'maxiter': 500, 'disp': False})
        
        if time.time() - start_time > time_limit:
            return best_points, best_r
            
        new_points = res.x.reshape(-1, 2)
        new_r, valid = compute_radius(new_points)
        
        if valid and new_r > best_r:
            best_points = new_points
            best_r = new_r
            
    except Exception:
        pass
        
    return best_points, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_points = None
    best_r = 0.0
    
    # 1. Hexagonal Lattice Initialization
    for spacing in [0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.20]:
        if time.time() > deadline - 5: break
        
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        
        if len(points) >= N:
            indices = np.argsort(radius)
            init_pts = points[indices[:N]]
            
            opt_pts, r = optimize_pack(init_pts, time_limit=5.0)
            if r > best_r:
                best_r = r
                best_points = opt_pts

    # 2. Ring Initialization
    ring_configs = [
        [0, 0.2, 0.4, 0.6, 0.8],
        [0, 0.25, 0.5, 0.75],
        [0, 0.3, 0.6, 0.9]
    ]
    
    for radii in ring_configs:
        if time.time() > deadline - 5: break
        
        points_list = []
        points_list.append([0.0, 0.0])
        
        for r in radii:
            if r == 0: continue
            n_circles = int(6 * (r / 0.2))
            if n_circles < 1: n_circles = 1
            angles = np.linspace(0, 2*np.pi, n_circles, endpoint=False)
            for a in angles:
                points_list.append([r * np.cos(a), r * np.sin(a)])
                
        points = np.array(points_list)
        
        if len(points) > N:
            idx = np.random.choice(len(points), N, replace=False)
            init_pts = points[idx]
        elif len(points) < N:
            extra_n = N - len(points)
            extra_pts = np.random.uniform(-1, 1, (extra_n, 2))
            norms = np.linalg.norm(extra_pts, axis=1)
            norms[norms == 0] = 1
            extra_pts = extra_pts / norms[:, np.newaxis] * 0.8
            init_pts = np.vstack([points, extra_pts])
        else:
            init_pts = points
            
        opt_pts, r = optimize_pack(init_pts, time_limit=5.0)
        if r > best_r:
            best_r = r
            best_points = opt_pts

    # 3. Random Initialization
    for i in range(15):
        if time.time() > deadline - 10: break
        
        angles = np.random.uniform(0, 2*np.pi, N)
        rads = np.sqrt(np.random.uniform(0, 1, N))
        init_pts = np.column_stack([rads * np.cos(angles), rads * np.sin(angles)])
        
        opt_pts, r = optimize_pack(init_pts, time_limit=5.0)
        if r > best_r:
            best_r = r
            best_points = opt_pts

    # 4. Polish
    if best_points is not None and time.time() < deadline - 10:
        for i in range(10):
            if time.time() > deadline - 5: break
            
            noisy_pts = best_points + np.random.normal(0, 0.02, best_points.shape)
            norms = np.linalg.norm(noisy_pts, axis=1)
            mask = norms > 0.98
            noisy_pts[mask] = noisy_pts[mask] * (0.98 / norms[mask])[:, np.newaxis]
            
            opt_pts, r = optimize_pack(noisy_pts, time_limit=5.0)
            if r > best_r:
                best_r = r
                best_points = opt_pts

    if best_points is None:
        # Fallback
        spacing = 0.25
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points = points[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98

    # Final validation
    r, valid = compute_radius(best_points)
    if not valid:
        # Return fallback
        spacing = 0.25
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points = points[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98

    return best_points
