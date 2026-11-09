import numpy as np
import scipy.optimize as opt
import time

def _compute_vertex_scores_fast(polygon):
    n = len(polygon)
    scores = np.zeros(n)
    
    for i in range(n):
        v_i = polygon[i]
        
        dists_L = np.zeros(9)
        dists_R = np.zeros(9)
        
        for k in range(1, 10):
            idx_R = (i + k) % n
            idx_L = (i - k) % n
            diff_R = polygon[idx_R] - v_i
            diff_L = polygon[idx_L] - v_i
            dists_R[k-1] = np.linalg.norm(diff_R)
            dists_L[k-1] = np.linalg.norm(diff_L)
            
        idx_M = (i + 10) % n
        diff_M = polygon[idx_M] - v_i
        dist_M = np.linalg.norm(diff_M)
        
        merged = np.empty(18)
        iL = 0
        iR = 0
        for m in range(18):
            if iL < 9 and (iR >= 9 or dists_L[iL] <= dists_R[iR]):
                merged[m] = dists_L[iL]
                iL += 1
            else:
                merged[m] = dists_R[iR]
                iR += 1
                
        pos = np.searchsorted(merged, dist_M)
        if pos < 18:
            merged = np.insert(merged, pos, dist_M)
        else:
            merged = np.append(merged, dist_M)
            
        d_sorted = merged
        
        if d_sorted[0] < 0.05:
            scores[i] = 0.0
            continue
            
        gaps = np.diff(d_sorted)
        k = np.argmin(gaps)
        
        D = (d_sorted[k] + d_sorted[k+1]) / 2.0
        
        if D < 0.1:
            scores[i] = 0.0
            continue
            
        if k == 0:
            selected = d_sorted[0:4]
        elif k == 17:
            selected = d_sorted[15:19]
        else:
            selected = d_sorted[k-1:k+3]
            
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        
        scores[i] = min(1.0, 1.0 / raw)
        
    return scores

def _min_score(polygon):
    return np.min(_compute_vertex_scores_fast(polygon))

def _convexity_penalty(polygon):
    n = len(polygon)
    penalty = 0.0
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i+1)%n]
        p2 = polygon[(i+2)%n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if cross < 1e-6:
            penalty += (1e-6 - cross) ** 2
    return penalty

def _objective(params):
    N = 20
    polygon = params.reshape(N, 2)
    
    centroid = np.mean(polygon, axis=0)
    polygon_norm = polygon - centroid
    dists_to_centroid = np.linalg.norm(polygon_norm, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist < 0.01:
        return 1e6
        
    polygon_norm = polygon_norm / mean_dist
    
    pen = _convexity_penalty(polygon_norm)
    if pen > 1e-3:
        return 10.0 + pen * 100.0
        
    score = _min_score(polygon_norm)
    
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    x0 = np.stack([x, y], axis=1).flatten()
    
    start_time = time.time()
    time_limit = 580.0
    
    best_params = x0.copy()
    best_score = _min_score(best_params.reshape(n, 2))
    
    bounds = [(-5, 5)] * (2*n)
    
    try:
        de_result = opt.differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=10,
            tol=1e-6,
            polish=False
        )
        
        if de_result.fun < -best_score:
            best_params = de_result.x
            best_score = -de_result.fun
            
    except Exception as e:
        pass
        
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        polygon = best_params.reshape(n, 2)
        centroid = np.mean(polygon, axis=0)
        polygon = polygon - centroid
        dists = np.linalg.norm(polygon, axis=1)
        mean_dist = np.mean(dists)
        if mean_dist > 0:
            polygon = polygon / mean_dist
        return polygon
        
    try:
        nm_result = opt.minimize(
            _objective, 
            best_params, 
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if nm_result.fun < -best_score:
            best_params = nm_result.x
            best_score = -nm_result.fun
            
    except Exception as e:
        pass
        
    polygon = best_params.reshape(n, 2)
    centroid = np.mean(polygon, axis=0)
    polygon = polygon - centroid
    dists = np.linalg.norm(polygon, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist > 0:
        polygon = polygon / mean_dist
        
    return polygon
