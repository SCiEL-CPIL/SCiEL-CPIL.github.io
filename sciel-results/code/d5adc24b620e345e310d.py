# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0

    # 1. Analytic/Parameterized Search for Rhombic Dodecahedron-like structure
    # Structure: 8 points at (±x, ±x, ±x) and 6 points at (±y, 0, 0), (0, ±y, 0), (0, 0, ±y)
    
    def get_points_rhombic(x, y):
        pts = []
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    pts.append([sx*x, sy*x, sz*x])
        pts.append([y, 0, 0])
        pts.append([-y, 0, 0])
        pts.append([0, y, 0])
        pts.append([0, -y, 0])
        pts.append([0, 0, y])
        pts.append([0, 0, -y])
        return np.array(pts)

    def calc_ratio(pts):
        # Calculate pairwise distances
        # n=14, 91 pairs
        d = np.zeros((14, 14))
        for i in range(14):
            for j in range(i+1, 14):
                dist = np.linalg.norm(pts[i] - pts[j])
                d[i, j] = dist
                d[j, i] = dist
        
        # Mask diagonal
        np.fill_diagonal(d, np.inf)
        
        d_min = np.min(d)
        d_max = np.max(d)
        
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def neg_log_ratio(params):
        x, y = params
        if x <= 0 or y <= 0:
            return 1e10
        pts = get_points_rhombic(x, y)
        r = calc_ratio(pts)
        if r <= 0:
            return 1e10
        return -np.log(r)

    # Optimize x, y
    best_score = 0.0
    best_points = None
    
    # Initial guess from x=1, y=2
    x0, y0 = 1.0, 2.0
    res = minimize(neg_log_ratio, [x0, y0], method='Nelder-Mead', 
                   options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000})
    
    if res.success or res.fun < 1e9:
        x_opt, y_opt = res.x
        pts_opt = get_points_rhombic(x_opt, y_opt)
        score_opt = calc_ratio(pts_opt)
        
        if score_opt > best_score:
            best_score = score_opt
            best_points = pts_opt

    # 2. General Optimization with Symmetric Initialization
    # We can try to optimize all 42 coordinates starting from the best symmetric solution
    # or random solutions, using a robust optimizer.
    # Since we have time, let's do a few random restarts of general optimization.
    
    def general_neg_log_ratio(coords_flat):
        pts = coords_flat.reshape(14, 3)
        # Center the points to remove translational invariance
        pts = pts - np.mean(pts, axis=0)
        # Scale to unit variance to remove scale invariance? 
        # No, the ratio is scale invariant. 
        # But if we don't fix scale, the optimizer might drift.
        # Let's fix the sum of squared norms to 1.
        norm_sq = np.sum(pts**2)
        if norm_sq < 1e-12:
            return 1e10
        pts = pts / np.sqrt(norm_sq)
        
        d = np.zeros((14, 14))
        for i in range(14):
            for j in range(i+1, 14):
                dist = np.linalg.norm(pts[i] - pts[j])
                d[i, j] = dist
                d[j, i] = dist
        np.fill_diagonal(d, np.inf)
        
        d_min = np.min(d)
        d_max = np.max(d)
        
        if d_max < 1e-12:
            return 1e10
        r = (d_min / d_max) ** 2
        if r <= 0:
            return 1e10
        return -np.log(r)

    # Use the best symmetric points as a starting point for general optimization
    if best_points is not None:
        # Normalize best_points
        bp = best_points - np.mean(best_points, axis=0)
        norm_sq = np.sum(bp**2)
        bp = bp / np.sqrt(norm_sq)
        
        res_gen = minimize(general_neg_log_ratio, bp.flatten(), method='Nelder-Mead',
                           options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000})
        
        if res_gen.success or res_gen.fun < 1e9:
            pts_gen = res_gen.x.reshape(14, 3)
            pts_gen = pts_gen - np.mean(pts_gen, axis=0)
            score_gen = calc_ratio(pts_gen)
            
            if score_gen > best_score:
                best_score = score_gen
                best_points = pts_gen

    # Try a few random starts
    for i in range(5):
        if time.time() > deadline - safety_margin:
            break
            
        np.random.seed(i+100)
        rand_pts = np.random.randn(14, 3)
        rand_pts = rand_pts - np.mean(rand_pts, axis=0)
        norm_sq = np.sum(rand_pts**2)
        rand_pts = rand_pts / np.sqrt(norm_sq)
        
        res_rand = minimize(general_neg_log_ratio, rand_pts.flatten(), method='Nelder-Mead',
                            options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 2000})
        
        if res_rand.success or res_rand.fun < 1e9:
            pts_rand = res_rand.x.reshape(14, 3)
            pts_rand = pts_rand - np.mean(pts_rand, axis=0)
            score_rand = calc_ratio(pts_rand)
            
            if score_rand > best_score:
                best_score = score_rand
                best_points = pts_rand

    # Final check
    if best_points is None:
        # Fallback to the analytic solution if optimization failed
        x_opt, y_opt = 1.0, 2.0
        best_points = get_points_rhombic(x_opt, y_opt)
    else:
        # Ensure points are finite and valid
        if not np.all(np.isfinite(best_points)):
            x_opt, y_opt = 1.0, 2.0
            best_points = get_points_rhombic(x_opt, y_opt)

    return best_points

# EVOLVE-BLOCK-END
