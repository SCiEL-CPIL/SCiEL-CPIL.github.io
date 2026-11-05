import numpy as np
import time
from scipy.optimize import minimize

def compute_log_score(points):
    """
    Compute the log of the score Delta/n^n for n=8.
    Delta is the product of ordered pair distances after scaling to diameter 2.
    Score = Delta / 8^8.
    We maximize log(Score) = log(Delta) - 8*log(8).
    log(Delta) = sum_{i != j} log(d_{ij} / diameter)
    = sum_{i != j} log(d_{ij}) - 8*7*log(diameter)
    Note: There are 8*7 = 56 ordered pairs.
    """
    points = np.array(points)
    n = len(points)
    
    # Compute pairwise distances
    # diff[i, j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Diameter is the max distance
    diameter = np.max(dists)
    if diameter <= 0:
        return -np.inf
        
    # Mask diagonal to avoid log(0)
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dists[mask]
    
    if np.any(valid_dists <= 0):
        return -np.inf
        
    # log(Delta) = sum(log(valid_dists)) - n*(n-1)*log(diameter)
    # Wait, the definition says: "Delta is the product over ordered pairs of distances after scaling the set to diameter 2"
    # Scaled distance d' = d / (diameter/2) = 2d/diameter
    # So Delta = prod_{i!=j} (2 * d_{ij} / diameter)
    # log(Delta) = sum_{i!=j} [log(2) + log(d_{ij}) - log(diameter)]
    # log(Delta) = n*(n-1)*log(2) + sum_{i!=j} log(d_{ij}) - n*(n-1)*log(diameter)
    
    n_pairs = n * (n - 1)
    log_delta = n_pairs * np.log(2) + np.sum(np.log(valid_dists)) - n_pairs * np.log(diameter)
    
    # Score = Delta / n^n
    # log(Score) = log(Delta) - n*log(n)
    log_score = log_delta - n * np.log(n)
    
    return log_score

def compute_smooth_log_score(points, beta=50.0):
    """
    Compute a smooth approximation of log score using soft-max for diameter.
    This makes the objective smoother for gradient-based optimization.
    """
    points = np.array(points)
    n = len(points)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dists[mask]
    
    if np.any(valid_dists <= 0):
        return -1e10
    
    # Soft-max approximation of log(max(d))
    # log(max(d)) ≈ (1/beta) * log(sum(exp(beta * d)))
    # But we need to be careful with numerical stability
    max_d = np.max(valid_dists)
    if max_d <= 0:
        return -1e10
    
    # Soft-max: log(sum(exp(beta * (d - max_d)))) / beta + max_d
    shifted = beta * (valid_dists - max_d)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    smooth_log_diameter = max_d + log_sum_exp / beta
    
    n_pairs = n * (n - 1)
    log_delta = n_pairs * np.log(2) + np.sum(np.log(valid_dists)) - n_pairs * smooth_log_diameter
    
    log_score = log_delta - n * np.log(n)
    
    return log_score

def negative_smooth_log_score(points_flat, beta=50.0):
    """
    Objective function for minimization using smooth surrogate.
    """
    points = points_flat.reshape(8, 2)
    log_score = compute_smooth_log_score(points, beta)
    if not np.isfinite(log_score):
        return 1e10
    return -log_score

def negative_log_score(points_flat):
    """
    Objective function for minimization.
    Minimizes -log_score.
    """
    points = points_flat.reshape(8, 2)
    log_score = compute_log_score(points)
    if not np.isfinite(log_score):
        return 1e10
    return -log_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_score = 1.0  # Regular octagon score
    # Regular octagon points
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    rng = np.random.default_rng(42)
    
    # List of initializations to try
    initializations = []
    
    # 1. Regular octagon
    initializations.append(best_points.copy())
    
    # 2. Two concentric squares - more varied radii
    square_angles_1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    square_angles_2 = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
    for r1 in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
        for r2 in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
            if abs(r1 - r2) < 0.1: continue
            p1 = np.column_stack([r1 * np.cos(square_angles_1), r1 * np.sin(square_angles_1)])
            p2 = np.column_stack([r2 * np.cos(square_angles_2), r2 * np.sin(square_angles_2)])
            initializations.append(np.vstack([p1, p2]))
    
    # 3. Perturbed regular octagons - more variations
    for _ in range(30):
        angles_pert = angles + rng.normal(0, 0.3, 8)
        radii_pert = np.ones(8) + rng.normal(0, 0.3, 8)
        radii_pert = np.abs(radii_pert) + 0.1  # Ensure positive
        p = np.column_stack([radii_pert * np.cos(angles_pert), radii_pert * np.sin(angles_pert)])
        initializations.append(p)
        
    # 4. Random points
    for _ in range(30):
        radii = rng.uniform(0.1, 1.8, 8)
        angles_r = rng.uniform(0, 2 * np.pi, 8)
        p = np.column_stack([radii * np.cos(angles_r), radii * np.sin(angles_r)])
        initializations.append(p)
    
    # 5. Antipodal pairs configuration
    # 4 pairs of points, each pair nearly antipodal
    for _ in range(15):
        base_angles = rng.uniform(0, np.pi/2, 4)
        radii = rng.uniform(0.5, 1.5, 4)
        pts = []
        for i in range(4):
            a = base_angles[i]
            r = radii[i]
            pts.append([r * np.cos(a), r * np.sin(a)])
            pts.append([-r * np.cos(a), -r * np.sin(a)])
        initializations.append(np.array(pts))
    
    # 6. Rectangular grid perturbations
    for _ in range(15):
        xs = rng.uniform(-1, 1, 8)
        ys = rng.uniform(-1, 1, 8)
        p = np.column_stack([xs, ys])
        initializations.append(p)
    
    # Sort initializations by their initial score to prioritize good starts
    scored_inits = []
    for init_p in initializations:
        if time.time() > deadline - 60:
            break
        try:
            ls = compute_log_score(init_p)
            if np.isfinite(ls):
                scored_inits.append((ls, init_p))
        except:
            pass
    
    # Sort by score descending
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    # Try top initializations with multiple methods
    for ls, init_p in scored_inits[:25]:
        if time.time() > deadline - 30:
            break
            
        x0 = init_p.flatten()
        
        # Method 1: Nelder-Mead
        try:
            res = minimize(negative_log_score, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.success or res.fun < 1e10:
                optimal_points = res.x.reshape(8, 2)
                log_score_val = -res.fun
                if np.isfinite(log_score_val):
                    score = np.exp(log_score_val)
                    if score > best_score:
                        best_score = score
                        best_points = optimal_points
        except Exception:
            pass
        
        # Method 2: COBYLA (can handle the non-smoothness better sometimes)
        if time.time() < deadline - 25:
            try:
                res2 = minimize(negative_log_score, x0, method='COBYLA', 
                               options={'maxiter': 3000, 'rhobeg': 0.1})
                if res2.fun < 1e10:
                    optimal_points2 = res2.x.reshape(8, 2)
                    log_score_val2 = -res2.fun
                    if np.isfinite(log_score_val2):
                        score2 = np.exp(log_score_val2)
                        if score2 > best_score:
                            best_score = score2
                            best_points = optimal_points2
            except Exception:
                pass
        
        # Method 3: Smooth surrogate with L-BFGS-B
        if time.time() < deadline - 20:
            try:
                res3 = minimize(negative_smooth_log_score, x0, method='L-BFGS-B', 
                               args=(50.0,),
                               options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-8})
                if res3.fun < 1e10:
                    optimal_points3 = res3.x.reshape(8, 2)
                    # Evaluate with the true (non-smooth) objective
                    log_score_val3 = compute_log_score(optimal_points3)
                    if np.isfinite(log_score_val3):
                        score3 = np.exp(log_score_val3)
                        if score3 > best_score:
                            best_score = score3
                            best_points = optimal_points3
            except Exception:
                pass

    # Refine the best point found with multiple methods
    if time.time() < deadline - 15:
        x_best = best_points.flatten()
        
        # High-precision Nelder-Mead
        try:
            res_refined = minimize(negative_log_score, x_best, method='Nelder-Mead', 
                           options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res_refined.success or res_refined.fun < 1e10:
                refined_points = res_refined.x.reshape(8, 2)
                log_score_val = -res_refined.fun
                if np.isfinite(log_score_val):
                    score = np.exp(log_score_val)
                    if score > best_score:
                        best_score = score
                        best_points = refined_points
        except Exception:
            pass
        
        # Smooth surrogate polish
        if time.time() < deadline - 10:
            try:
                for beta in [100.0, 200.0, 500.0]:
                    res_smooth = minimize(negative_smooth_log_score, best_points.flatten(), 
                                         method='L-BFGS-B', args=(beta,),
                                         options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-10})
                    if res_smooth.fun < 1e10:
                        smooth_pts = res_smooth.x.reshape(8, 2)
                        log_score_val = compute_log_score(smooth_pts)
                        if np.isfinite(log_score_val):
                            score = np.exp(log_score_val)
                            if score > best_score:
                                best_score = score
                                best_points = smooth_pts
            except Exception:
                pass

    # Scale best_points to diameter 2 for final return
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    diameter = np.max(dists)
    if diameter > 0:
        best_points = best_points * (2.0 / diameter)
        
    return best_points.tolist()

if __name__ == "__main__":
    pts = construct()
    # Calculate final score for debugging
    print(f"Best score: {np.exp(compute_log_score(np.array(pts)))}")
