import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def compute_min_triangle_area(points):
    """Compute minimum triangle area among all triples."""
    n = len(points)
    if n < 3:
        return 0.0
    
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
    Compute the soft minimum of an array of areas using log-sum-exp trick.
    soft_min(x) = min_val - 1/tau * log(sum(exp(-tau * (x - min_val))))
    """
    min_val = np.min(areas)
    shifted = areas - min_val
    # Clip to avoid overflow in exp
    shifted = np.maximum(shifted, -50.0 / tau)
    log_sum = np.log(np.sum(np.exp(-tau * shifted)))
    return min_val - (1.0 / tau) * log_sum

def compute_all_areas(points):
    """Compute all triangle areas for n points."""
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
    
    return areas

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    N = 13
    best_score = 0.0
    best_points = None
    
    def time_left():
        return deadline - time.time()
    
    # Strategy: Optimize angles of points on the unit circle.
    # This assumes the optimal configuration is convex and close to circular.
    # We fix point 0 at angle 0. The other N-1 points have free angles.
    # We must ensure the angles are sorted to maintain the convex hull order.
    # To handle sorting, we optimize the differences between consecutive angles.
    # Let d_i be the angle difference between point i and i-1.
    # Sum(d_i) = 2 * pi.
    # We can optimize d_1 ... d_{N-2} and set d_{N-1} = 2*pi - sum(d_1...d_{N-2}).
    # This ensures positivity if we constrain d_i > 0.
    
    def angles_to_points(angles):
        """Convert angles to points on unit circle."""
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    def params_to_points(params):
        """
        Convert params (angle differences) to points.
        params shape: (N-2,)
        """
        d = params
        # Ensure positive differences
        d = np.maximum(d, 1e-6)
        # The last difference is determined by the constraint sum = 2*pi
        # We need N-1 differences for N points (cyclic).
        # Let's define N-1 differences d_0 ... d_{N-2}.
        # Wait, we have N points. Let angles be a_0, ..., a_{N-1}.
        # a_0 = 0.
        # a_1 = d_0
        # a_2 = d_0 + d_1
        # ...
        # a_{N-1} = sum(d_0...d_{N-2})
        # The gap from a_{N-1} to 2*pi is d_{N-1}.
        # We optimize d_0 ... d_{N-2}.
        # d_{N-1} = 2*pi - sum(d_0...d_{N-2}).
        # We must ensure d_{N-1} > 0.
        
        sum_d = np.sum(d)
        if sum_d >= 2 * np.pi:
            # Scale down to ensure the last gap is positive
            scale = (2 * np.pi - 1e-6) / sum_d
            d = d * scale
        
        angles = np.zeros(N)
        angles[0] = 0.0
        angles[1:] = np.cumsum(d)
        # The last angle is < 2*pi. The implicit gap is 2*pi - angles[-1].
        
        return angles_to_points(angles)
    
    def points_to_params(pts):
        """Convert points to angle differences."""
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        angles = np.sort(angles)
        # Ensure angles are in [0, 2*pi)
        angles = angles % (2 * np.pi)
        angles = np.sort(angles)
        
        # Calculate differences
        diffs = np.diff(np.concatenate([angles, [angles[0] + 2 * np.pi]]))
        # We optimize the first N-2 differences. The last one is derived.
        return diffs[:N-2]
    
    def objective(params):
        """Minimize negative soft-min score."""
        try:
            pts = params_to_points(params)
            areas = compute_all_areas(pts)
            sm = soft_min(areas, 50.0)  # Fixed tau for speed, or pass as arg
            
            try:
                hull = ConvexHull(pts)
                hull_area = hull.volume
                if hull_area < 1e-10:
                    return 1e6
            except:
                return 1e6
            
            score = sm / hull_area
            return -score
        except:
            return 1e6

    # Initial guess: Regular N-gon
    # Angles: 0, 2pi/N, 4pi/N, ...
    # Differences: all 2pi/N
    initial_params = np.full(N-2, 2 * np.pi / N)
    
    # Run optimization
    # We will run a few iterations with increasing tau to sharpen the minimum
    taus = [20, 50, 100, 200]
    
    current_params = initial_params.copy()
    
    for tau in taus:
        if time_left() < 10.0:
            break
            
        def obj_tau(params):
            try:
                pts = params_to_points(params)
                areas = compute_all_areas(pts)
                sm = soft_min(areas, tau)
                try:
                    hull = ConvexHull(pts)
                    hull_area = hull.volume
                    if hull_area < 1e-10:
                        return 1e6
                except:
                    return 1e6
                score = sm / hull_area
                return -score
            except:
                return 1e6

        try:
            res = minimize(obj_tau, current_params, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12})
            if res.fun < 1e5:
                current_params = res.x
                # Update best score with exact calculation
                pts = params_to_points(current_params)
                s = get_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
        except:
            pass

    # Local search with perturbations on the best parameters found
    if best_points is not None and time_left() > 20.0:
        x_best = points_to_params(best_points)
        
        for pert_scale in [0.01, 0.05, 0.1]:
            if time_left() < 10.0:
                break
            for _ in range(10):
                if time_left() < 5.0:
                    break
                x_pert = x_best + np.random.normal(0, pert_scale, len(x_best))
                # Clamp to positive
                x_pert = np.maximum(x_pert, 1e-6)
                
                def obj_final(params):
                    try:
                        pts = params_to_points(params)
                        areas = compute_all_areas(pts)
                        sm = soft_min(areas, 200.0)
                        try:
                            hull = ConvexHull(pts)
                            hull_area = hull.volume
                            if hull_area < 1e-10:
                                return 1e6
                        except:
                            return 1e6
                        score = sm / hull_area
                        return -score
                    except:
                        return 1e6

                try:
                    res_p = minimize(obj_final, x_pert, method='L-BFGS-B',
                                     options={'maxiter': 100, 'ftol': 1e-12})
                    if res_p.fun < 1e5:
                        pts = params_to_points(res_p.x)
                        s = get_score(pts)
                        if s > best_score:
                            best_score = s
                            best_points = pts.copy()
                            x_best = res_p.x
                except:
                    pass

    # If no points found (should not happen), fallback to regular polygon
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = angles_to_points(angles)
        best_score = get_score(best_points)

    return best_points
