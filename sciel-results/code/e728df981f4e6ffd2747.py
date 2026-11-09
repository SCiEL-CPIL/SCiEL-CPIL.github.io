import numpy as np
import time
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    
    def compute_score(pts):
        pts = np.array(pts, dtype=float)
        centroid = pts.mean(axis=0)
        pts_norm = pts - centroid
        dists_to_centroid = np.linalg.norm(pts_norm, axis=1)
        mean_dist = dists_to_centroid.mean()
        if mean_dist <= 0.01:
            return 0.0
        pts_norm = pts_norm / mean_dist
        
        # Check convexity
        # Calculate cross products of consecutive edges
        min_score = 1.0
        valid = True
        for i in range(n):
            p0 = pts_norm[i]
            p1 = pts_norm[(i+1) % n]
            p2 = pts_norm[(i+2) % n]
            cross = (p1[0]-p0[0])*(p2[1]-p1[1]) - (p1[1]-p0[1])*(p2[0]-p1[0])
            if cross < 1e-8: # Must be strictly convex, so cross > 0 (assuming CCW)
                valid = False
                break
        
        if not valid:
            return 0.0
            
        for i in range(n):
            v = pts_norm[i]
            d = np.linalg.norm(pts_norm - v, axis=1)
            d_sorted = np.sort(d)
            d_other = d_sorted[1:]
            
            if d_other[0] < 0.05:
                return 0.0
            
            gaps = np.diff(d_other)
            k = np.argmin(gaps)
            D = (d_other[k] + d_other[k+1]) / 2
            if D < 0.1:
                return 0.0
            
            if k == 0:
                d_four = d_other[0:4]
            elif k == 17:
                d_four = d_other[15:19]
            else:
                d_four = d_other[k-1:k+3]
            
            raw = np.mean(np.maximum(d_four/D, D/d_four))
            score = min(1.0, 1.0/raw)
            min_score = min(min_score, score)
            
        return min_score

    def objective(params):
        # params: 5 radii, 5 angles (relative to sector start)
        r = params[:5]
        delta = params[5:]
        
        # Ensure sorted angles
        delta_sorted = np.sort(delta)
        
        pts = np.zeros((20, 2))
        for j in range(5):
            base_angle = 2 * np.pi * j / 5
            for m in range(5):
                idx = 5 * j + m
                angle = base_angle + delta_sorted[m]
                pts[idx, 0] = r[m] * np.cos(angle)
                pts[idx, 1] = r[m] * np.sin(angle)
                
        # Check if points are in cyclic order
        # The construction above places them in cyclic order if delta_sorted is sorted
        # and within [0, 2pi/5)
        
        score = compute_score(pts)
        # We want to maximize score, so minimize negative score
        return -score

    # Initial guess: Regular 20-gon
    # r = [1, 1, 1, 1, 1]
    # delta = [0, pi/10, 2pi/10, 3pi/10, 4pi/10]
    init_r = np.ones(5)
    init_delta = np.array([0, np.pi/10, 2*np.pi/10, 3*np.pi/10, 4*np.pi/10])
    init_params = np.concatenate([init_r, init_delta])
    
    # Bounds
    bounds = [(0.1, 5.0)] * 5 + [(0.0, 2*np.pi/5)] * 5
    
    start_time = time.time()
    deadline = 630.0
    
    best_params = init_params.copy()
    best_score = compute_score(np.zeros((20,2))) # Placeholder, will be updated
    
    # Run optimization
    try:
        res = minimize(objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if res.fun < objective(best_params):
            best_params = res.x
            best_score = -res.fun
    except Exception as e:
        pass
    
    # If we have time, try a few random restarts
    if time.time() < deadline - 60:
        for _ in range(10):
            if time.time() > deadline - 30:
                break
            # Random perturbation of initial guess
            r_rand = 1.0 + 0.5 * np.random.randn(5)
            r_rand = np.clip(r_rand, 0.1, 5.0)
            delta_rand = np.sort(np.random.uniform(0, 2*np.pi/5, 5))
            params_rand = np.concatenate([r_rand, delta_rand])
            
            res = minimize(objective, params_rand, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            
            if res.fun < objective(best_params):
                best_params = res.x
                best_score = -res.fun

    # Construct final points
    r = best_params[:5]
    delta = np.sort(best_params[5:])
    
    final_pts = np.zeros((20, 2))
    for j in range(5):
        base_angle = 2 * np.pi * j / 5
        for m in range(5):
            idx = 5 * j + m
            angle = base_angle + delta[m]
            final_pts[idx, 0] = r[m] * np.cos(angle)
            final_pts[idx, 1] = r[m] * np.sin(angle)
            
    return final_pts
