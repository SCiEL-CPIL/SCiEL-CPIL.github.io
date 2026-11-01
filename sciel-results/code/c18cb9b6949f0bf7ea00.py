import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# Precompute triangle indices once
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """Compute min triangle area / convex hull area, vectorized."""
    n = len(points)
    if n < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def neg_score(x):
    points = x.reshape((N, 2))
    return -compute_score(points)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(12345)
    
    def try_update(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    # --- Strategy: Hybrid Global Search (DE on structured configs) + Local Refinement ---
    
    # Configuration 1: Two concentric rings (7 outer, 7 inner)
    def params_to_points_2ring(params):
        r_out = 1.0
        r_in = params[0]
        phi = params[1]
        
        angles_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
        angles_in = angles_out + phi
        
        pts_out = np.column_stack((r_out * np.cos(angles_out), r_out * np.sin(angles_out)))
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        
        return np.vstack((pts_out, pts_in))
    
    def neg_score_2ring(params):
        pts = params_to_points_2ring(params)
        return -compute_score(pts)
    
    bounds_2ring = [(0.0, 1.0), (0, 2*np.pi/7)]
    
    # Configuration 2: Three rings (5 outer, 5 middle, 4 inner)
    def params_to_points_3ring(params):
        r_out = 1.0
        r_mid = params[0]
        r_in = params[1]
        phi_mid = params[2]
        phi_in = params[3]
        
        angles_out = np.linspace(0, 2*np.pi, 5, endpoint=False)
        angles_mid = angles_out + phi_mid
        angles_in = np.linspace(0, 2*np.pi, 4, endpoint=False) + phi_in
        
        pts_out = np.column_stack((r_out * np.cos(angles_out), r_out * np.sin(angles_out)))
        pts_mid = np.column_stack((r_mid * np.cos(angles_mid), r_mid * np.sin(angles_mid)))
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        
        return np.vstack((pts_out, pts_mid, pts_in))
    
    def neg_score_3ring(params):
        pts = params_to_points_3ring(params)
        return -compute_score(pts)
    
    bounds_3ring = [(0.0, 1.0), (0.0, 1.0), (0, 2*np.pi/5), (0, 2*np.pi/4)]
    
    # Configuration 3: Four rings (4, 4, 3, 3) - New structural idea from inspiration
    def params_to_points_4ring(params):
        r1, r2, r3, r4 = params[0], params[1], params[2], params[3]
        th1, th2, th3, th4 = params[4], params[5], params[6], params[7]
        
        # Fixed counts: 4, 4, 3, 3
        n1, n2, n3, n4 = 4, 4, 3, 3
        
        pts1 = np.column_stack((r1*np.cos(th1 + np.linspace(0, 2*np.pi, n1, endpoint=False)), 
                                r1*np.sin(th1 + np.linspace(0, 2*np.pi, n1, endpoint=False))))
        pts2 = np.column_stack((r2*np.cos(th2 + np.linspace(0, 2*np.pi, n2, endpoint=False)), 
                                r2*np.sin(th2 + np.linspace(0, 2*np.pi, n2, endpoint=False))))
        pts3 = np.column_stack((r3*np.cos(th3 + np.linspace(0, 2*np.pi, n3, endpoint=False)), 
                                r3*np.sin(th3 + np.linspace(0, 2*np.pi, n3, endpoint=False))))
        pts4 = np.column_stack((r4*np.cos(th4 + np.linspace(0, 2*np.pi, n4, endpoint=False)), 
                                r4*np.sin(th4 + np.linspace(0, 2*np.pi, n4, endpoint=False))))
        
        return np.vstack((pts1, pts2, pts3, pts4))
    
    def neg_score_4ring(params):
        pts = params_to_points_4ring(params)
        return -compute_score(pts)
    
    bounds_4ring = [
        (0.5, 1.5),  # r1
        (0.3, 1.0),  # r2
        (0.1, 0.7),  # r3
        (0.0, 0.4),  # r4
        (0, 2*np.pi), # th1
        (0, 2*np.pi), # th2
        (0, 2*np.pi), # th3
        (0, 2*np.pi)  # th4
    ]
    
    # Configuration 4: Two rings with different counts (8 outer, 6 inner)
    def params_to_points_86(params):
        r_out = 1.0
        r_in = params[0]
        phi = params[1]
        
        angles_out = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles_in = np.linspace(0, 2*np.pi, 6, endpoint=False) + phi
        
        pts_out = np.column_stack((r_out * np.cos(angles_out), r_out * np.sin(angles_out)))
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        
        return np.vstack((pts_out, pts_in))
    
    def neg_score_86(params):
        pts = params_to_points_86(params)
        return -compute_score(pts)
    
    bounds_86 = [(0.0, 1.0), (0, 2*np.pi/6)]

    # Run Differential Evolution on these low-dimensional spaces
    configs = [
        {"neg": neg_score_2ring, "bounds": bounds_2ring, "name": "2ring"},
        {"neg": neg_score_3ring, "bounds": bounds_3ring, "name": "3ring"},
        {"neg": neg_score_4ring, "bounds": bounds_4ring, "name": "4ring"},
        {"neg": neg_score_86, "bounds": bounds_86, "name": "86ring"}
    ]
    
    for cfg in configs:
        if time.time() - start_time > deadline - 25:
            break
            
        try:
            res = differential_evolution(
                cfg["neg"], 
                cfg["bounds"], 
                seed=42, 
                maxiter=80, 
                tol=1e-6,
                polish=True,
                init='sobol'
            )
            if res.success or res.fun < 0:
                # Reconstruct points from optimal params
                if cfg["name"] == "2ring":
                    pts = params_to_points_2ring(res.x)
                elif cfg["name"] == "3ring":
                    pts = params_to_points_3ring(res.x)
                elif cfg["name"] == "4ring":
                    pts = params_to_points_4ring(res.x)
                else:
                    pts = params_to_points_86(res.x)
                
                try_update(pts)
        except Exception as e:
            pass

    # Also try a few random initializations with local refinement as a backup
    inits = []
    
    # Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    inits.append(p_reg)
    
    # Jittered regular polygon
    for _ in range(3):
        inits.append(p_reg + rng.normal(0, 0.1, (N, 2)))
    
    # Random in disk
    for _ in range(5):
        r = np.sqrt(rng.random(N))
        th = rng.random(N) * 2 * np.pi
        inits.append(np.column_stack((r*np.cos(th), r*np.sin(th))))
        
    for p in inits:
        if time.time() - start_time > deadline - 15:
            break
        try_update(p)

    # Final Phase: Local refinement of the best point set found so far
    if best_points is not None and time.time() - start_time < deadline - 5:
        x0 = best_points.flatten()
        # Use Nelder-Mead for fine tuning
        try:
            res = minimize(
                neg_score, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            try_update(res.x.reshape((N, 2)))
        except Exception:
            pass
        
        # One more Powell if time permits
        if time.time() - start_time < deadline - 2:
            x0 = best_points.flatten()
            try:
                res = minimize(
                    neg_score, 
                    x0, 
                    method='Powell',
                    options={'maxiter': 500, 'xtol': 1e-9, 'ftol': 1e-11}
                )
                try_update(res.x.reshape((N, 2)))
            except Exception:
                pass

    if best_points is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
