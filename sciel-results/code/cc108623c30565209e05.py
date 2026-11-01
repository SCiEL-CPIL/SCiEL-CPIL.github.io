import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

def compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
        
    # Compute all triangle areas
    # There are N*(N-1)*(N-2)/6 triangles
    n = len(points)
    min_area = float('inf')
    
    # Optimization: Use vectorized calculation if possible, but N=14 is small
    # Loop is fine: 364 triangles
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Area of triangle with vertices p_i, p_j, p_k
                # 0.5 * | (x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i) |
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                area = 0.5 * abs(dx1 * dy2 - dx2 * dy1)
                if area < min_area:
                    min_area = area
                    if min_area < 1e-12:
                        return 0.0
                        
    return min_area / hull_area

def neg_score(x):
    """
    Objective function for minimization (we maximize score).
    x is a flat array of length 2*N.
    """
    points = x.reshape((N, 2))
    score = compute_score(points)
    return -score

def _two_ring_points(r, theta):
    """
    Generate 14 points: 7 on outer ring (radius 1.0), 7 on inner ring (radius r).
    Inner ring is rotated by theta.
    """
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = angles_outer + theta
    
    p_outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    p_inner = np.column_stack((r * np.cos(angles_inner), r * np.sin(angles_inner)))
    
    points = np.vstack((p_outer, p_inner))
    return points

def _neg_ring_score(params):
    """
    Objective function for the 2-ring parameterization.
    params: [r, theta]
    """
    r, theta = params
    if r < 0.0 or r > 1.0:
        return 1e9
    points = _two_ring_points(r, theta)
    score = compute_score(points)
    return -score

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # 1. Global search over the symmetric 2-ring family (r, theta)
    # r in [0, 1], theta in [0, 2*pi/7] due to symmetry of heptagon
    bounds = [(0.0, 1.0), (0.0, 2 * np.pi / 7)]
    
    try:
        # Differential Evolution is a global optimizer, suitable for this low-dim space
        res_de = differential_evolution(
            _neg_ring_score, 
            bounds, 
            seed=42, 
            maxiter=200, 
            tol=1e-8,
            polish=True,
            init='sobol'
        )
        
        if res_de.success or res_de.fun < 0:
            r_opt, theta_opt = res_de.x
            points_sym = _two_ring_points(r_opt, theta_opt)
            score_sym = compute_score(points_sym)
            
            if score_sym > best_score:
                best_score = score_sym
                best_points = points_sym.copy()
    except Exception:
        pass

    # Fallback or alternative: Grid search if DE fails or is slow
    if best_points is None:
        rng = np.random.default_rng(42)
        # Coarse grid
        r_vals = np.linspace(0.2, 0.9, 10)
        th_vals = np.linspace(0, 2*np.pi/7, 10)
        for r in r_vals:
            for th in th_vals:
                points = _two_ring_points(r, th)
                s = compute_score(points)
                if s > best_score:
                    best_score = s
                    best_points = points.copy()
    
    # 2. Local refinement in full 28D space from the best symmetric start
    if best_points is not None:
        x0 = best_points.flatten()
        
        # Use Nelder-Mead for local refinement
        # This allows breaking symmetry slightly if it improves the score
        try:
            res_nm = minimize(neg_score, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res_nm.success:
                new_points = res_nm.x.reshape((N, 2))
                new_score = compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
        except Exception:
            pass
        
        # Try Powell as well
        if time.time() - start_time < deadline - 5:
            try:
                res_pw = minimize(neg_score, x0, method='Powell', 
                            options={'maxiter': 1000, 'xtol': 1e-10, 'ftol': 1e-12})
                if res_pw.success:
                    new_points = res_pw.x.reshape((N, 2))
                    new_score = compute_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
            except Exception:
                pass

    # 3. If we still have time and haven't improved much, try a few random starts 
    #    and local optimize, just in case the symmetric assumption is too strong
    if time.time() - start_time < deadline - 20:
        rng = np.random.default_rng(123)
        for _ in range(3):
            # Random points in unit disk
            r = np.sqrt(rng.random(N))
            theta = rng.random(N) * 2 * np.pi
            p_rand = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
            
            # Quick local optimize
            try:
                res_r = minimize(neg_score, p_rand.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
                new_points = res_r.x.reshape((N, 2))
                new_score = compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
            except Exception:
                pass
            if time.time() - start_time > deadline - 5:
                break

    return best_points
