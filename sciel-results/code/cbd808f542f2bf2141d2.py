import numpy as np
from scipy.optimize import minimize, differential_evolution
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n = 8
    
    def compute_score(points):
        """
        Compute Delta/n^n where Delta is product of all ordered pair distances
        after scaling to diameter 2.
        """
        pts = np.asarray(points, dtype=float)
        if len(pts) != n:
            return -np.inf
        
        # Check for distinct points
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-10:
                    return -np.inf
        
        # Compute diameter
        max_d = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                max_d = max(max_d, d)
        
        if max_d < 1e-15:
            return -np.inf
        
        # Scale to diameter 2
        scale = 2.0 / max_d
        pts_scaled = pts * scale
        
        # Compute log of product of all ordered pair distances
        # Ordered pairs means (i,j) and (j,i) both count, so it's 2 * sum over i<j of log(d_ij)
        log_prod = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts_scaled[i] - pts_scaled[j])
                if d < 1e-15:
                    return -np.inf
                log_prod += math.log(d)
        
        log_prod_ordered = 2.0 * log_prod
        
        # Delta = exp(log_prod_ordered), score = Delta / n^n
        # log(score) = log_prod_ordered - n*log(n)
        log_score = log_prod_ordered - n * math.log(n)
        
        return math.exp(log_score)
    
    def negative_log_score(points):
        """Objective for minimization: -log(score)"""
        pts = np.asarray(points, dtype=float)
        if len(pts) != n:
            return 1e10
        
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-10:
                    return 1e10
        
        max_d = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                max_d = max(max_d, d)
        
        if max_d < 1e-15:
            return 1e10
        
        scale = 2.0 / max_d
        pts_scaled = pts * scale
        
        log_prod = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts_scaled[i] - pts_scaled[j])
                if d < 1e-15:
                    return 1e10
                log_prod += math.log(d)
        
        log_prod_ordered = 2.0 * log_prod
        log_score = log_prod_ordered - n * math.log(n)
        
        return -log_score
    
    # Regular octagon as baseline
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    reg_score = compute_score(reg_pts)
    
    best_pts = reg_pts.copy()
    best_score = reg_score
    
    # Symmetric parameterization: 4 pairs of antipodal points
    # Each pair is (r_i * cos(theta_i), r_i * sin(theta_i)) and (-r_i * cos(theta_i), -r_i * sin(theta_i))
    # This ensures point i and point i+4 are antipodal
    # We optimize 4 radii and 4 angles = 8 parameters
    
    def symmetric_objective(params):
        if time.time() > deadline:
            return 1e10
        
        radii = params[:4]
        thetas = params[4:]
        
        pts = np.zeros((n, 2))
        for i in range(4):
            r = radii[i]
            theta = thetas[i]
            pts[i] = [r * math.cos(theta), r * math.sin(theta)]
            pts[i + 4] = [-r * math.cos(theta), -r * math.sin(theta)]
        
        # Compute score
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                    return 1e10
        
        max_d = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                max_d = max(max_d, d)
        
        if max_d < 1e-15:
            return 1e10
        
        scale = 2.0 / max_d
        pts_scaled = pts * scale
        
        log_prod = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts_scaled[i] - pts_scaled[j])
                if d < 1e-15:
                    return 1e10
                log_prod += math.log(d)
        
        log_prod_ordered = 2.0 * log_prod
        log_score = log_prod_ordered - n * math.log(n)
        
        return -log_score
    
    # Initial guess: regular octagon in symmetric form
    # Angles: 0, pi/4, pi/2, 3pi/4, and their antipodes
    x0_sym = np.array([1.0, 1.0, 1.0, 1.0,
                        0.0, np.pi/4, np.pi/2, 3.0*np.pi/4])
    
    # Try symmetric optimization
    np.random.seed(42)
    num_sym_restarts = 15
    
    for i in range(num_sym_restarts):
        if time.time() > deadline - 5:
            break
        
        if i == 0:
            x0 = x0_sym.copy()
        else:
            noise_r = np.random.normal(0, 0.1, size=4)
            noise_t = np.random.normal(0, 0.15, size=4)
            x0 = x0_sym.copy()
            x0[:4] += noise_r
            x0[4:] += noise_t
        
        try:
            res = minimize(symmetric_objective, x0, method='L-BFGS-B',
                          bounds=[(0.01, 10.0)] * 4 + [(-2*np.pi, 2*np.pi)] * 4,
                          options={'maxiter': 300, 'ftol': 1e-14})
            
            if np.all(np.isfinite(res.x)):
                radii = res.x[:4]
                thetas = res.x[4:]
                pts = np.zeros((n, 2))
                for j in range(4):
                    r = radii[j]
                    theta = thetas[j]
                    pts[j] = [r * math.cos(theta), r * math.sin(theta)]
                    pts[j + 4] = [-r * math.cos(theta), -r * math.sin(theta)]
                
                s = compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
        except Exception:
            pass
    
    # Now try full 16D optimization from the best symmetric result
    flat_best = best_pts.flatten()
    
    for i in range(10):
        if time.time() > deadline - 5:
            break
        
        if i == 0:
            x0 = flat_best.copy()
        else:
            noise = np.random.normal(0, 0.1, size=16)
            x0 = flat_best + noise
        
        try:
            res = minimize(negative_log_score, x0, method='L-BFGS-B',
                          bounds=[(-10.0, 10.0)] * 16,
                          options={'maxiter': 200, 'ftol': 1e-14})
            
            if np.all(np.isfinite(res.x)):
                pts = res.x.reshape(n, 2)
                s = compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
        except Exception:
            pass
    
    # Try differential evolution on symmetric params if time allows
    if time.time() < deadline - 20:
        try:
            bounds_de = [(0.01, 10.0)] * 4 + [(-2*np.pi, 2*np.pi)] * 4
            res_de = differential_evolution(
                symmetric_objective, bounds_de,
                maxiter=30, popsize=15, tol=1e-6,
                seed=42, polish=False
            )
            
            if np.all(np.isfinite(res_de.x)):
                radii = res_de.x[:4]
                thetas = res_de.x[4:]
                pts = np.zeros((n, 2))
                for j in range(4):
                    r = radii[j]
                    theta = thetas[j]
                    pts[j] = [r * math.cos(theta), r * math.sin(theta)]
                    pts[j + 4] = [-r * math.cos(theta), -r * math.sin(theta)]
                
                s = compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
                    
                    # Polish
                    if time.time() < deadline - 5:
                        x0_polish = np.concatenate([radii, thetas])
                        res_polish = minimize(symmetric_objective, x0_polish, method='L-BFGS-B',
                                            bounds=bounds_de, options={'maxiter': 200, 'ftol': 1e-14})
                        if np.all(np.isfinite(res_polish.x)):
                            radii_p = res_polish.x[:4]
                            thetas_p = res_polish.x[4:]
                            pts_p = np.zeros((n, 2))
                            for j in range(4):
                                r = radii_p[j]
                                theta = thetas_p[j]
                                pts_p[j] = [r * math.cos(theta), r * math.sin(theta)]
                                pts_p[j + 4] = [-r * math.cos(theta), -r * math.sin(theta)]
                            s_p = compute_score(pts_p)
                            if s_p > best_score:
                                best_score = s_p
                                best_pts = pts_p.copy()
        except Exception:
            pass
    
    # Final validation
    best_pts = np.asarray(best_pts, dtype=float)
    diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    
    if np.min(d) < 1e-6 or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts
        best_score = reg_score
    
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5
    
    return best_pts.tolist()
