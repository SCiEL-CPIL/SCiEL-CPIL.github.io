import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

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
    
    # Precompute all C(n,3) triangle indices
    idx = np.array([(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)])
    p_i = points[idx[:, 0]]
    p_j = points[idx[:, 1]]
    p_k = points[idx[:, 2]]
    
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
    deadline = 145.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -float('inf')
    
    def try_update(points):
        nonlocal best_points, best_score
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s

    # --- Strategy: Search symmetric parameterized configurations ---
    
    # Configuration 1: Two concentric rings (7 outer, 7 inner)
    # Params: [r_out, r_in, phi_offset]
    # r_out is fixed to 1.0 for normalization invariance? No, scale doesn't matter for ratio.
    # Let r_out = 1.0. r_in in (0, 1). phi in (0, 2*pi/7).
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
    # Params: [r_mid, r_in, phi_mid, phi_in]
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
    
    # Configuration 3: Two rings with different counts? e.g. 8 outer, 6 inner.
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
    # DE is robust for global optimization in low dims.
    
    configs = [
        {"neg": neg_score_2ring, "bounds": bounds_2ring, "name": "2ring"},
        {"neg": neg_score_3ring, "bounds": bounds_3ring, "name": "3ring"},
        {"neg": neg_score_86, "bounds": bounds_86, "name": "86ring"}
    ]
    
    for cfg in configs:
        if time.time() - start_time > deadline - 20:
            break
            
        try:
            res = differential_evolution(
                cfg["neg"], 
                cfg["bounds"], 
                seed=42, 
                maxiter=100, 
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
                else:
                    pts = params_to_points_86(res.x)
                
                try_update(pts)
        except Exception as e:
            pass

    # Also try a few random initializations with local refinement as a backup
    rng = np.random.default_rng(12345)
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
        if time.time() - start_time > deadline - 10:
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
