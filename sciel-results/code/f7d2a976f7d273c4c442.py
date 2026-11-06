import numpy as np
from scipy.optimize import minimize
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
            return 0.0
        
        # Check for distinct points
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-10:
                    return 0.0
        
        # Compute diameter
        max_d = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                max_d = max(max_d, d)
        
        if max_d < 1e-15:
            return 0.0
        
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
                    return 0.0
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

    # 1. Baseline: Regular Octagon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = compute_score(reg_pts)
    best_pts = reg_pts.copy()

    # 2. Targeted Search: Two-Square Configuration (D4 Symmetry)
    # Points: 
    # Square 1 (radius r1): angles 0, pi/2, pi, 3pi/2
    # Square 2 (radius r2): angles pi/4, 3pi/4, 5pi/4, 7pi/4
    # This is a subset of the antipodal pair symmetry.
    
    def two_square_points(r1, r2):
        pts = np.zeros((n, 2))
        # Square 1
        pts[0] = [r1, 0]
        pts[1] = [0, r1]
        pts[2] = [-r1, 0]
        pts[3] = [0, -r1]
        # Square 2
        c = r2 / math.sqrt(2)
        pts[4] = [c, c]
        pts[5] = [-c, c]
        pts[6] = [-c, -c]
        pts[7] = [c, -c]
        return pts

    def two_square_objective(params):
        r1, r2 = params
        if r1 <= 0 or r2 <= 0:
            return 1e10
        pts = two_square_points(r1, r2)
        return negative_log_score(pts)

    # Optimize r1, r2
    # The regular octagon corresponds to r1 = r2 = 1 (scaled).
    # The optimum is known to be around r1/r2 != 1.
    # Let's try a few starting points for r1, r2.
    
    initial_guesses = [
        [1.0, 1.0],      # Regular octagon
        [1.2, 0.8],      # Perturbed
        [0.8, 1.2],      # Perturbed
        [1.5, 0.5],      # More extreme
        [0.5, 1.5],      # More extreme
        [1.1, 0.9],      # Slight perturbation
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(two_square_objective, guess, method='Nelder-Mead', 
                           options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 2000})
            if res.success or res.fun < 1e9:
                r1_opt, r2_opt = res.x
                if r1_opt > 0 and r2_opt > 0:
                    cand_pts = two_square_points(r1_opt, r2_opt)
                    s_cand = compute_score(cand_pts)
                    if s_cand > best_score:
                        best_score = s_cand
                        best_pts = cand_pts.copy()
        except Exception:
            pass

    # 3. Full Space Refinement from Best Candidate
    # Use the best point found (likely the two-square config) to seed a full 16D optimization.
    # This allows the points to move off the D4 symmetry if that yields a higher score,
    # although D4 is expected to be the global max.
    
    if time.time() < deadline - 20:
        x0 = best_pts.flatten()
        # Add a tiny bit of noise to escape potential flat spots if we are exactly at a symmetric point
        # But actually, L-BFGS-B should handle symmetric points fine.
        
        # We will do a few restarts from the best point with different noise levels
        for noise_level in [0.0, 0.01, 0.05]:
            if time.time() > deadline - 5:
                break
            x_init = x0 + np.random.normal(0, noise_level, size=16)
            try:
                res = minimize(negative_log_score, x_init, method='L-BFGS-B',
                               bounds=[(-10.0, 10.0)] * 16,
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
                
                if np.all(np.isfinite(res.x)):
                    pts = res.x.reshape(n, 2)
                    s = compute_score(pts)
                    if s > best_score:
                        best_score = s
                        best_pts = pts.copy()
            except Exception:
                pass

    # 4. Final Validation and Formatting
    best_pts = np.asarray(best_pts, dtype=float)
    if not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts
        best_score = compute_score(reg_pts)
        
    # Check distinctness
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    if min_dist < 1e-6:
        best_pts = reg_pts
        best_score = compute_score(reg_pts)

    # Ensure coordinates are within bounds
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()
