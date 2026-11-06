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

    def two_square_log_score(r1, r2):
        """
        Analytical log score for the two-square configuration.
        Returns log(Delta / n^n)
        """
        if r1 <= 1e-10 or r2 <= 1e-10:
            return -1e10
            
        # Distances:
        # Within Sq1: 4 sides (r1*sqrt(2)), 2 diags (2*r1)
        # Within Sq2: 4 sides (r2*sqrt(2)), 2 diags (2*r2)
        # Between: 4 short (sqrt(r1^2 + r2^2 - sqrt(2)r1r2)), 4 long (sqrt(r1^2 + r2^2 + sqrt(2)r1r2))
        
        # Log product of unordered distances:
        # 4*log(r1*sqrt(2)) + 2*log(2*r1) + 4*log(r2*sqrt(2)) + 2*log(2*r2)
        # + 4*log(d_short) + 4*log(d_long)
        
        log_d_short_sq = 0.5 * math.log(r1**2 + r2**2 - math.sqrt(2)*r1*r2)
        log_d_long_sq = 0.5 * math.log(r1**2 + r2**2 + math.sqrt(2)*r1*r2)
        
        log_prod_unordered = (
            4 * (math.log(r1) + 0.5*math.log(2)) +
            2 * (math.log(2) + math.log(r1)) +
            4 * (math.log(r2) + 0.5*math.log(2)) +
            2 * (math.log(2) + math.log(r2)) +
            4 * log_d_short_sq +
            4 * log_d_long_sq
        )
        
        # Simplified:
        # 6*log(r1) + 6*log(r2) + 8*log(2) + 2*log(d_short_sq) + 2*log(d_long_sq)
        # d_short_sq * d_long_sq = r1^4 + r2^4
        # So 2*log(d_short_sq) + 2*log(d_long_sq) = 2*log(r1^4 + r2^4)
        
        log_prod_unordered = 6*math.log(r1) + 6*math.log(r2) + 8*math.log(2) + 2*math.log(r1**4 + r2**4)
        
        # Diameter D
        d1 = 2 * r1
        d2 = 2 * r2
        d3 = math.sqrt(r1**2 + r2**2 + math.sqrt(2)*r1*r2)
        D = max(d1, d2, d3)
        
        # Score = exp(2 * log_prod_unordered - 28 * log(D) - 8 * log(8))
        log_score = 2 * log_prod_unordered - 28 * math.log(D) - 8 * math.log(8)
        
        return log_score

    def neg_two_square_log_score(params):
        r1, r2 = params
        if r1 <= 0 or r2 <= 0:
            return 1e10
        return -two_square_log_score(r1, r2)

    # Optimize r1, r2
    # We know regular octagon is r1=r2=1. Score is 1.0.
    # Let's search for better r1, r2.
    
    best_r1 = 1.0
    best_r2 = 1.0
    best_sym_score = two_square_log_score(1.0, 1.0)
    
    # Grid search to find promising regions
    r_range = np.linspace(0.1, 2.0, 40)
    for r1 in r_range:
        if time.time() > deadline - 10:
            break
        for r2 in r_range:
            s = two_square_log_score(r1, r2)
            if s > best_sym_score:
                best_sym_score = s
                best_r1 = r1
                best_r2 = r2

    # Local refinement around the best grid point
    if time.time() < deadline - 5:
        try:
            res = minimize(neg_two_square_log_score, [best_r1, best_r2], method='Nelder-Mead', 
                           options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 2000})
            if res.success or res.fun < 1e9:
                r1_opt, r2_opt = res.x
                if r1_opt > 0 and r2_opt > 0:
                    s_opt = two_square_log_score(r1_opt, r2_opt)
                    if s_opt > best_sym_score:
                        best_sym_score = s_opt
                        best_r1 = r1_opt
                        best_r2 = r2_opt
        except Exception:
            pass

    # Also try starting from regular octagon specifically
    try:
        res_reg = minimize(neg_two_square_log_score, [1.0, 1.0], method='Nelder-Mead', 
                           options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 2000})
        if res_reg.success or res_reg.fun < 1e9:
            r1_reg, r2_reg = res_reg.x
            if r1_reg > 0 and r2_reg > 0:
                s_reg = two_square_log_score(r1_reg, r2_reg)
                if s_reg > best_sym_score:
                    best_sym_score = s_reg
                    best_r1 = r1_reg
                    best_r2 = r2_reg
    except Exception:
        pass

    # Generate candidate from optimized symmetric model
    cand_pts = two_square_points(best_r1, best_r2)
    s_cand = compute_score(cand_pts)
    if s_cand > best_score:
        best_score = s_cand
        best_pts = cand_pts.copy()

    # 3. Full Space Refinement from Best Candidate
    # Use the best point found to seed a full 16D optimization.
    # This allows the points to move off the D4 symmetry if that yields a higher score.
    
    if time.time() < deadline - 20:
        x0 = best_pts.flatten()
        
        # Add a tiny bit of noise to escape potential flat spots
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
