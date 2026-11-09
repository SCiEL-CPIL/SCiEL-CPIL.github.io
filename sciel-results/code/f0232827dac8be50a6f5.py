import numpy as np
from scipy.optimize import differential_evolution

def calculate_score(angles):
    n = len(angles)
    x = np.cos(angles)
    y = np.sin(angles)
    pts = np.stack([x, y], axis=1)
    
    # Normalize
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    dists = np.linalg.norm(pts, axis=1)
    mean_dist = np.mean(dists)
    if mean_dist <= 0.01:
        return 0.0
    pts = pts / mean_dist
    
    min_score = 1.0
    for i in range(n):
        vi = pts[i]
        dists_i = np.linalg.norm(pts - vi, axis=1)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        dists_i = dists_i[mask]
        
        if np.min(dists_i) < 0.05:
            return 0.0
        
        sorted_dists = np.sort(dists_i)
        gaps = np.diff(sorted_dists)
        k = np.argmin(gaps)
        D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
        
        if D < 0.1:
            return 0.0
            
        if k == 0:
            idxs = [0, 1, 2, 3]
        elif k == 17:
            idxs = [15, 16, 17, 18]
        else:
            idxs = [k-1, k, k+1, k+2]
            
        selected_dists = sorted_dists[idxs]
        ratios = np.maximum(selected_dists / D, D / selected_dists)
        raw = np.mean(ratios)
        score = min(1.0, 1.0 / raw)
        
        if score < min_score:
            min_score = score
            
    return min_score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    
    def objective(gaps):
        if np.any(gaps <= 1e-5):
            return 0.0
        
        total_sum = np.sum(gaps)
        if total_sum >= 2 * np.pi:
            return 0.0
            
        gap_20 = 2 * np.pi - total_sum
        if gap_20 <= 1e-5:
            return 0.0
            
        all_gaps = np.append(gaps, gap_20)
        
        angles = np.zeros(n)
        for i in range(1, n):
            angles[i] = np.sum(all_gaps[:i])
            
        return -calculate_score(angles)

    eps = 1e-4
    bounds = [(eps, 2*np.pi - 19*eps)] * 19
    
    best_gaps = np.full(19, 2.0 * np.pi / 20)
    best_score_val = 0.0
    
    try:
        res = differential_evolution(
            objective, 
            bounds=bounds, 
            maxiter=200, 
            tol=1e-5, 
            seed=42,
            polish=True,
            workers=1,
            popsize=10
        )
        
        if res.fun < 0:
            best_gaps = res.x
            best_score_val = -res.fun
            
        reg_gaps = np.full(19, 2.0 * np.pi / 20)
        reg_score = -objective(reg_gaps)
        if reg_score > best_score_val:
            best_gaps = reg_gaps
            best_score_val = reg_score
            
    except Exception as e:
        pass

    gap_20 = 2 * np.pi - np.sum(best_gaps)
    all_gaps = np.append(best_gaps, gap_20)
    
    angles = np.zeros(20)
    for i in range(1, 20):
        angles[i] = np.sum(all_gaps[:i])
        
    x = np.cos(angles)
    y = np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    return polygon
