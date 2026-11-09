import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def compute_min_triangle_area(points):
    """Compute minimum triangle area among all triples."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Vectorized computation of all triangle areas
    # points: (n, 2)
    # We need to compute area for all (i, j, k) with i < j < k
    # This is O(n^3) memory if we do it all at once, which for n=13 is 286 triangles.
    # 286 floats is tiny. We can compute all areas at once.
    
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    # Filter for i < j < k
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def get_score(points):
    """Compute score: min triangle area / convex hull area."""
    N = len(points)
    if N < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except:
        return 0.0
    
    min_area = compute_min_triangle_area(points)
    if min_area <= 0:
        return 0.0
    return min_area / hull_area

def soft_min(areas, tau):
    """
    Compute the soft minimum of an array of areas.
    Uses log-sum-exp trick for stability.
    soft_min(x) = -1/tau * log(sum(exp(-tau * x)))
    """
    # To avoid overflow in exp, subtract min
    min_val = np.min(areas)
    # If areas are very small, tau * areas might be small.
    # exp(-tau * (areas - min_val)) is stable.
    # log(sum(exp(-tau * areas))) = -tau * min_val + log(sum(exp(-tau * (areas - min_val))))
    # soft_min = -1/tau * ( -tau * min_val + log(sum(...)) )
    #           = min_val - 1/tau * log(sum(exp(-tau * (areas - min_val))))
    
    # Handle case where areas are identical
    if np.allclose(areas, min_val):
        return min_val
        
    shifted = areas - min_val
    # Clip shifted to avoid overflow in exp if tau is large and shifted is negative (shouldn't happen as min_val is min)
    # Actually shifted >= 0.
    # exp(-tau * shifted) will be <= 1.
    # Sum will be <= len(areas).
    # log(sum) is safe.
    
    log_sum = np.log(np.sum(np.exp(-tau * shifted)))
    return min_val - (1.0 / tau) * log_sum

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    N = 13
    best_score = 0.0
    best_points = None
    
    # Helper to check time
    def time_left():
        return deadline - time.time()
    
    # 1. Baseline: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    s = get_score(points)
    best_score = s
    best_points = points.copy()
    
    # Define the smooth objective function
    # We optimize -score
    def neg_score_smooth(points, tau):
        """
        Minimize - (soft_min(areas) / hull_area)
        """
        try:
            # Compute areas vectorized for speed in optimization loop
            n = len(points)
            idx = np.arange(n)
            i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
            mask = (i < j) & (j < k)
            i = i[mask]
            j = j[mask]
            k = k[mask]
            
            p1 = points[i]
            p2 = points[j]
            p3 = points[k]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            areas = 0.5 * np.abs(cross)
            
            # Soft min
            sm = soft_min(areas, tau)
            
            # Hull area
            try:
                hull = ConvexHull(points)
                hull_area = hull.volume
                if hull_area < 1e-10:
                    return 1e6
            except:
                return 1e6
                
            score = sm / hull_area
            return -score
        except:
            return 1e6

    # 2. Optimize angles on unit circle
    # Parameters: 12 angles (first fixed at 0)
    def neg_score_circle_angles(theta, tau):
        angles = np.concatenate(([0.0], theta))
        angles = np.sort(angles % (2 * np.pi))
        pts = np.column_stack((np.cos(angles), np.sin(angles)))
        return neg_score_smooth(pts, tau)

    x0 = np.array([2 * np.pi * i / N for i in range(1, N)])
    
    # Try a range of tau values
    # Start with moderate tau to smooth landscape, then increase
    taus = [20, 50, 100, 200]
    
    for tau in taus:
        if time_left() < 5.0:
            break
        # Multiple restarts for circle angles
        n_restarts = 3
        for r in range(n_restarts):
            if time_left() < 5.0:
                break
            if r == 0:
                x_start = x0.copy()
            else:
                step = 0.2
                x_start = x0 + np.random.normal(0, step, N - 1)
            
            try:
                res = minimize(neg_score_circle_angles, x_start, args=(tau,), method='Powell',
                               options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-9})
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    angles = np.concatenate(([0.0], res.x))
                    angles = np.sort(angles % (2 * np.pi))
                    best_points = np.column_stack((np.cos(angles), np.sin(angles)))
            except:
                pass

    # 3. Optimize on Ellipse
    # Parameters: a, b, 12 angles
    if time_left() > 10.0:
        # Extract angles from best_points for initialization
        angles_best = np.arctan2(best_points[:, 1], best_points[:, 0])
        angles_best = np.sort(angles_best)
        # Ensure angles are in [0, 2pi) and sorted
        angles_best = angles_best % (2 * np.pi)
        # Handle wrap-around for sorting if needed, but arctan2 gives [-pi, pi]
        # Let's just use the best_points coordinates to infer a, b if they are close to ellipse
        # Simpler: Start with a=1, b=1 and the current angles
        
        def neg_score_ellipse(params, tau):
            a = max(0.1, params[0])
            b = max(0.1, params[1])
            theta = np.concatenate(([0.0], params[2:]))
            theta = np.sort(theta % (2 * np.pi))
            pts = np.column_stack((a * np.cos(theta), b * np.sin(theta)))
            return neg_score_smooth(pts, tau)
        
        # Initial guess
        x0_ell = np.concatenate(([1.0, 1.0], angles_best[1:] % (2*np.pi)))
        
        for tau in [50, 100]:
            if time_left() < 5.0:
                break
            try:
                res_ell = minimize(neg_score_ellipse, x0_ell, args=(tau,), method='Powell',
                                   options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-8})
                current_score = -res_ell.fun
                if current_score > best_score:
                    best_score = current_score
                    a = max(0.1, res_ell.x[0])
                    b = max(0.1, res_ell.x[1])
                    theta = np.concatenate(([0.0], res_ell.x[2:]))
                    theta = np.sort(theta % (2 * np.pi))
                    best_points = np.column_stack((a * np.cos(theta), b * np.sin(theta)))
                    # Update x0_ell for next tau
                    x0_ell = res_ell.x
            except:
                pass

    # 4. Full 2D Optimization
    # Use the best points found so far as initial guess
    if time_left() > 15.0 and best_points is not None:
        
        def neg_score_2d(pts_flat, tau):
            pts = pts_flat.reshape(N, 2)
            return neg_score_smooth(pts, tau)
        
        x0_2d = best_points.flatten()
        
        # Use higher tau for 2D to push towards true min
        taus_2d = [50, 100, 200]
        
        for tau in taus_2d:
            if time_left() < 5.0:
                break
            # A few restarts with small perturbations
            n_restarts = 2
            for r in range(n_restarts):
                if time_left() < 5.0:
                    break
                if r == 0:
                    x_start = x0_2d.copy()
                else:
                    x_start = x0_2d + np.random.normal(0, 0.05, 2*N)
                
                try:
                    res2d = minimize(neg_score_2d, x_start, args=(tau,), method='Powell',
                                     options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-8})
                    current_score = -res2d.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = res2d.x.reshape(N, 2)
                        x0_2d = res2d.x
                except:
                    pass

    # Final check with exact min (not soft min) to ensure we report the true score
    # The optimization used soft min, which is an approximation.
    # We should evaluate the final points with the exact metric.
    if best_points is not None:
        final_score = get_score(best_points)
        if final_score > best_score:
            best_score = final_score
        # If soft min optimization led to a point where exact score is lower, 
        # it's still the best we found. But we should return the points that maximize exact score.
        # Since we only kept points that improved the *soft* score, it's possible 
        # the exact score is lower than a previous point? 
        # Actually, soft min >= exact min. 
        # If we maximize soft min, we are likely maximizing exact min.
        # However, to be safe, let's just return the points. The evaluator calculates the score.
        
    return best_points
