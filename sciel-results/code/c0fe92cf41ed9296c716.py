import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import random
import math

def score_from_points(P):
    """
    Calculate the score for a set of points P (8x2).
    Score = Delta / n^n, where Delta is product of ordered pair distances,
    after scaling P to diameter 2.
    """
    n = P.shape[0]
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    dists_sq = dists_sq[mask]
    
    if np.any(dists_sq < 1e-12):
        return 0.0
        
    dists = np.sqrt(dists_sq)
    D = np.max(dists)
    
    if D < 1e-12:
        return 0.0
        
    # Scale to diameter 2
    scale = 2.0 / D
    # Delta is product of all ordered pairs (i != j)
    # log(Delta) = sum(log(dists))
    # We want Delta_scaled / n^n
    # Delta_scaled = Delta * scale^(n*(n-1))
    # log(Score) = log(Delta) + n*(n-1)*log(scale) - n*log(n)
    
    sum_log_dists = np.sum(np.log(dists))
    log_score = sum_log_dists + n*(n-1)*np.log(scale) - n*np.log(n)
    
    return np.exp(log_score)

def objective_d2_sym(x):
    """
    x: 8 parameters [x1, y1, x2, y2, x3, y3, x4, y4]
    Constructs 8 points with D2 symmetry:
    (x1, y1), (x1, -y1), (-x1, y1), (-x1, -y1)
    (x2, y2), (x2, -y2), (-x2, y2), (-x2, -y2)
    ...
    Returns negative log score (for minimization) or just negative score.
    Using log score is more stable.
    """
    P = np.array([
        [x[0],  x[1]], [x[0], -x[1]], [-x[0],  x[1]], [-x[0], -x[1]],
        [x[2],  x[3]], [x[2], -x[3]], [-x[2],  x[3]], [-x[2], -x[3]],
        [x[4],  x[5]], [x[4], -x[5]], [-x[4],  x[5]], [-x[4], -x[5]],
        [x[6],  x[7]], [x[6], -x[7]], [-x[6],  x[7]], [-x[6], -x[7]]
    ])
    
    # Ensure distinct points
    if np.any(np.sum(np.diff(P, axis=0)**2, axis=1) < 1e-12): # simple check, might miss non-adjacent
         pass
         
    # Check for duplicates
    dists_sq = np.sum((P[:, np.newaxis, :] - P[np.newaxis, :, :])**2, axis=2)
    if np.any(dists_sq[~np.eye(8, dtype=bool)] < 1e-12):
        return 1e100
        
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    mask = ~np.eye(8, dtype=bool)
    dists_sq = dists_sq[mask]
    
    if np.any(dists_sq < 1e-12):
        return 1e100
        
    dists = np.sqrt(dists_sq)
    D = np.max(dists)
    
    if D < 1e-12:
        return 1e100
        
    sum_log_dists = np.sum(np.log(dists))
    n = 8
    log_score = sum_log_dists + n*(n-1)*np.log(2.0/D) - n*np.log(n)
    
    return -log_score

def objective_full(x):
    """
    x: 16 parameters [x1, y1, ..., x8, y8]
    Returns negative log score.
    """
    P = x.reshape(8, 2)
    dists_sq = np.sum((P[:, np.newaxis, :] - P[np.newaxis, :, :])**2, axis=2)
    mask = ~np.eye(8, dtype=bool)
    dists_sq = dists_sq[mask]
    
    if np.any(dists_sq < 1e-12):
        return 1e100
        
    dists = np.sqrt(dists_sq)
    D = np.max(dists)
    
    if D < 1e-12:
        return 1e100
        
    sum_log_dists = np.sum(np.log(dists))
    n = 8
    log_score = sum_log_dists + n*(n-1)*np.log(2.0/D) - n*np.log(n)
    
    return -log_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_val = -1e100
    best_points = None
    
    # Initialize with regular octagon
    base_angles = np.arange(8) * (2 * np.pi / 8)
    P_reg = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    val_reg = score_from_points(P_reg)
    best_val = val_reg
    best_points = P_reg.copy()
    
    # Strategy 1: Search in D2 symmetric space using Differential Evolution
    # Variables: 4 pairs of (x, y)
    # Bounds: Since diameter is 2, points must be within [-1, 1] roughly, but let's be safe [-2, 2]
    bounds = [(-2.0, 2.0)] * 8
    
    # Use Differential Evolution for global search in reduced space
    try:
        if time.time() < deadline - 30:
            res_de = differential_evolution(
                objective_d2_sym, 
                bounds, 
                seed=42, 
                maxiter=200, 
                tol=1e-8,
                polish=True,
                workers=1
            )
            if res_de.fun < 1e99:
                val_de = -res_de.fun
                # Reconstruct points to get actual score
                x_de = res_de.x
                P_de = np.array([
                    [x_de[0],  x_de[1]], [x_de[0], -x_de[1]], [-x_de[0],  x_de[1]], [-x_de[0], -x_de[1]],
                    [x_de[2],  x_de[3]], [x_de[2], -x_de[3]], [-x_de[2],  x_de[3]], [-x_de[2], -x_de[3]],
                    [x_de[4],  x_de[5]], [x_de[4], -x_de[5]], [-x_de[4],  x_de[5]], [-x_de[4], -x_de[5]],
                    [x_de[6],  x_de[7]], [x_de[6], -x_de[7]], [-x_de[6],  x_de[7]], [-x_de[6], -x_de[7]]
                ])
                val_de_actual = score_from_points(P_de)
                if val_de_actual > best_val:
                    best_val = val_de_actual
                    best_points = P_de
    except Exception:
        pass
    
    # Strategy 2: Full 2D search with Differential Evolution (more expensive, but less constrained)
    # Only do this if we have time and D2 didn't find something huge
    if time.time() < deadline - 40:
        bounds_full = [(-2.0, 2.0)] * 16
        try:
            # Use a smaller maxiter for full space
            res_de_full = differential_evolution(
                objective_full, 
                bounds_full, 
                seed=42, 
                maxiter=100, 
                tol=1e-7,
                polish=False, # Polish later
                workers=1
            )
            if res_de_full.fun < 1e99:
                P_full = res_de_full.x.reshape(8, 2)
                # Polish the result
                res_polish = minimize(
                    objective_full, 
                    P_full.reshape(-1), 
                    method='Powell', 
                    options={'maxiter': 200, 'ftol': 1e-10, 'xtol': 1e-10}
                )
                if res_polish.fun < 1e99:
                    P_polished = res_polish.x.reshape(8, 2)
                    val_polished = score_from_points(P_polished)
                    if val_polished > best_val:
                        best_val = val_polished
                        best_points = P_polished
        except Exception:
            pass
            
    # Strategy 3: Refine the best found so far with high-precision Powell
    if time.time() < deadline - 5:
        try:
            res_final = minimize(
                objective_full, 
                best_points.reshape(-1), 
                method='Powell', 
                options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-12}
            )
            if res_final.fun < 1e99:
                P_final = res_final.x.reshape(8, 2)
                val_final = score_from_points(P_final)
                if val_final > best_val:
                    best_val = val_final
                    best_points = P_final
        except Exception:
            pass
            
    # Final safety check: ensure distinct points and valid coordinates
    if best_points is None:
        return P_reg.tolist()
        
    # Check for duplicates one last time
    dists_sq = np.sum((best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :])**2, axis=2)
    if np.any(dists_sq[~np.eye(8, dtype=bool)] < 1e-9):
        return P_reg.tolist()
        
    return best_points.tolist()
