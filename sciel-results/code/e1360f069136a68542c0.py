import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Generate candidate clusters
    def axial_to_cart(q, r):
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        return np.array([x, y])

    def grow_cluster(n_hex):
        pts = [np.array([0.0, 0.0])]
        axial = [(0,0)]
        neighbors = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
        
        while len(pts) < n_hex:
            best_pt = None
            best_score = np.inf
            candidates = set()
            for q, r in axial:
                for dq, dr in neighbors:
                    nq, nr = q+dq, r+dr
                    if (nq, nr) not in axial:
                        candidates.add((nq, nr))
            
            for q, r in candidates:
                pt = axial_to_cart(q, r)
                if len(pts) > 0:
                    centroid = np.mean(pts, axis=0)
                else:
                    centroid = np.array([0,0])
                
                new_pts = pts + [pt]
                new_centroid = np.mean(new_pts, axis=0)
                
                max_dist = max(np.linalg.norm(p - new_centroid) for p in new_pts)
                
                if max_dist < best_score:
                    best_score = max_dist
                    best_pt = pt
                    best_axial = (q, r)
            
            if best_pt is not None:
                pts.append(best_pt)
                axial.append(best_axial)
            else:
                break
        return np.array(pts)

    clusters = []
    c1 = grow_cluster(N)
    # Center the cluster
    c1 = c1 - np.mean(c1, axis=0)
    clusters.append(c1)
    
    # Add a second cluster with a different growth bias?
    # For now, just one good cluster is likely sufficient if optimized well.
    # But let's add a "line" just in case, though it won't win.
    # Actually, let's try a slightly different growth: minimize max distance to origin (not centroid).
    # This might produce a different shape.
    
    def grow_cluster_origin(n_hex):
        pts = [np.array([0.0, 0.0])]
        axial = [(0,0)]
        neighbors = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
        
        while len(pts) < n_hex:
            best_pt = None
            best_score = np.inf
            candidates = set()
            for q, r in axial:
                for dq, dr in neighbors:
                    nq, nr = q+dq, r+dr
                    if (nq, nr) not in axial:
                        candidates.add((nq, nr))
            
            for q, r in candidates:
                pt = axial_to_cart(q, r)
                # Score: max distance from origin for all points in new set
                new_pts = pts + [pt]
                max_dist = max(np.linalg.norm(p) for p in new_pts)
                
                if max_dist < best_score:
                    best_score = max_dist
                    best_pt = pt
                    best_axial = (q, r)
            
            if best_pt is not None:
                pts.append(best_pt)
                axial.append(best_axial)
            else:
                break
        return np.array(pts)

    c2 = grow_cluster_origin(N)
    c2 = c2 - np.mean(c2, axis=0)
    # Avoid duplicate
    if not np.allclose(c1, c2):
        clusters.append(c2)

    # 2. Define objective
    NORMALS = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in [0, 60, 120, 180, 240, 300]
    ])

    def objective(params, cluster_pts):
        alpha, tx, ty = params
        rot = np.array([
            [np.cos(alpha), -np.sin(alpha)],
            [np.sin(alpha), np.cos(alpha)]
        ])
        pts = (cluster_pts @ rot.T) + np.array([tx, ty])
        projs = pts @ NORMALS.T
        max_abs_proj = np.max(np.abs(projs))
        S = 2.0 * max_abs_proj / np.sqrt(3)
        return S

    best_S = 40.0
    best_params = None
    best_cluster = None

    # 3. Optimize
    for cluster in clusters:
        if time.time() > deadline:
            break
            
        for alpha0 in np.linspace(0, np.pi/3, 13):
            if time.time() > deadline:
                break
                
            x0 = [alpha0, 0.0, 0.0]
            try:
                res = minimize(objective, x0, args=(cluster,), method='Nelder-Mead', 
                               options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})
                S = res.fun
                if S < best_S:
                    best_S = S
                    best_params = res.x
                    best_cluster = cluster
            except Exception:
                pass

    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    alpha, tx, ty = best_params
    rot = np.array([
        [np.cos(alpha), -np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)]
    ])
    
    inner = []
    for p in best_cluster:
        p_trans = (p @ rot.T) + np.array([tx, ty])
        inner.append([p_trans[0], p_trans[1], np.rad2deg(alpha)])

    return inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END
