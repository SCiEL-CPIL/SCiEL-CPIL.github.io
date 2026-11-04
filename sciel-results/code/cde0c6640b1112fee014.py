import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 14

# Precompute angles for 7-fold symmetry
# Outer ring: radius 1, angles 2*pi*k/7
# Inner ring: radius r, angles 2*pi*k/7 + theta

# We need to evaluate the minimum triangle area for the (7,7) configuration.
# There are 7 outer points O_0...O_6 and 7 inner points I_0...I_6.
# Total 14 points.
# Due to rotational symmetry, we can fix one point or just analyze relative indices.
# Let's define the points in a function that takes r and theta.

def _fast_score_ring(r, theta):
    """
    Calculate the minimum triangle area / hull area for two concentric regular heptagons.
    Outer: radius 1, angles 2*pi*k/7.
    Inner: radius r, angles 2*pi*k/7 + theta.
    """
    if r < 0 or r > 1:
        return 0.0
    
    # Hull is the convex hull of the outer points (since r <= 1).
    # Area of regular heptagon with circumradius 1:
    # A = 7 * 0.5 * 1^2 * sin(2*pi/7)
    hull_area = 3.5 * np.sin(2 * np.pi / 7)
    
    # Generate points
    angles_outer = np.arange(7) * 2 * np.pi / 7
    angles_inner = angles_outer + theta
    
    # Coordinates
    x_o = np.cos(angles_outer)
    y_o = np.sin(angles_outer)
    
    x_i = r * np.cos(angles_inner)
    y_i = r * np.sin(angles_inner)
    
    # We need to find the minimum area among all triples.
    # Triples can be:
    # 1. 3 Outer (OOO)
    # 2. 3 Inner (III)
    # 3. 2 Outer, 1 Inner (OOI)
    # 4. 1 Outer, 2 Inner (OII)
    
    # Case 1: OOO
    # All OOO triangles are congruent? No, there are two types in a heptagon:
    # (0,1,2) and (0,1,3).
    # Let's compute areas for representative relative indices.
    # Indices modulo 7.
    
    min_area = np.inf
    
    # Helper to compute area of triangle given 3 points (x1,y1), (x2,y2), (x3,y3)
    def area(p1, p2, p3):
        return 0.5 * np.abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))
    
    # We can vectorize this by checking all combinations of indices from the 14 points.
    # But 14 choose 3 is 364. That's very fast.
    # Let's just compute all 364 areas. It's safer and simpler.
    
    # Stack points
    # Order: O0..O6, I0..I6
    pts = np.vstack([
        np.column_stack((x_o, y_o)),
        np.column_stack((x_i, y_i))
    ])
    
    # Compute all triangle areas
    # Use broadcasting
    # i, j, k indices
    # To save memory/time, we can iterate or use a precomputed list of indices.
    # For N=14, 364 is small.
    
    # Precomputed indices for N=14
    indices = np.array([(i, j, k) for i in range(14) for j in range(i+1, 14) for k in range(j+1, 14)])
    
    p_i = pts[indices[:, 0]]
    p_j = pts[indices[:, 1]]
    p_k = pts[indices[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _neg_fast_score(params):
    r, theta = params
    # Theta is periodic with 2*pi/7 due to symmetry of the regular heptagon.
    # We can constrain theta to [0, 2*pi/7] to avoid periodicity issues in optimizer.
    # However, differential evolution or minimize might step outside.
    # Let's just evaluate as is, but maybe wrap theta?
    # Actually, the function is periodic. Let's just evaluate.
    return -_fast_score_ring(r, theta)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Focus on the (7,7) configuration as it is the most symmetric.
    # Initial guess
    # Literature suggests r ~ 0.5-0.7, theta ~ pi/7 or 0.
    
    best_score = 0.0
    best_params = None
    
    # 1. Coarse Grid Search
    # r in [0, 1], theta in [0, 2*pi/7]
    r_vals = np.linspace(0.1, 0.9, 20)
    theta_vals = np.linspace(0, 2*np.pi/7, 20)
    
    for r in r_vals:
        for theta in theta_vals:
            if time.time() > deadline - 10:
                break
            s = _fast_score_ring(r, theta)
            if s > best_score:
                best_score = s
                best_params = (r, theta)
        
    # 2. Local Optimization
    if best_params is not None:
        r0, theta0 = best_params
        # Bounds
        bounds = [(0.0, 1.0), (0.0, 2*np.pi/7)]
        
        # Use L-BFGS-B for smooth optimization
        try:
            res = minimize(
                _neg_fast_score,
                np.array(best_params),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if res.fun < 0: # i.e. score > 0
                s = -res.fun
                if s > best_score:
                    best_score = s
                    best_params = res.x
        except Exception:
            pass
        
        # Polish with Nelder-Mead for high precision
        if time.time() < deadline - 5:
            try:
                res_nm = minimize(
                    _neg_fast_score,
                    np.array(best_params),
                    method='Nelder-Mead',
                    options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 2000, 'adaptive': True}
                )
                if -res_nm.fun > best_score:
                    best_score = -res_nm.fun
                    best_params = res_nm.x
            except Exception:
                pass

    # 3. Construct the final points
    if best_params is None:
        # Fallback
        r, theta = 0.5, 0.0
    else:
        r, theta = best_params
        
    angles_outer = np.arange(7) * 2 * np.pi / 7
    angles_inner = angles_outer + theta
    
    x_o = np.cos(angles_outer)
    y_o = np.sin(angles_outer)
    
    x_i = r * np.cos(angles_inner)
    y_i = r * np.sin(angles_inner)
    
    points = np.vstack([
        np.column_stack((x_o, y_o)),
        np.column_stack((x_i, y_i))
    ])
    
    # Optional: Final polish on full coordinates?
    # The symmetry is likely exact or near-exact. Breaking symmetry might help if the optimum isn't perfectly symmetric,
    # but for N=14, 7-fold symmetry is a very strong attractor.
    # Given the time saved by the fast evaluator, we have plenty of time.
    # Let's try a quick polish on the 28 coordinates using the original _calculate_score.
    
    if time.time() < deadline - 2:
        # Define the original score function for full points
        TRIPLE_INDICES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
        
        def _full_score(pts):
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pts)
                hull_area = hull.volume
                if hull_area <= 1e-9: return 0.0
            except: return 0.0
            p_i = pts[TRIPLE_INDICES[:, 0]]
            p_j = pts[TRIPLE_INDICES[:, 1]]
            p_k = pts[TRIPLE_INDICES[:, 2]]
            cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
                    (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
            min_area = 0.5 * np.min(np.abs(cross))
            return min_area / hull_area
        
        def _neg_full_score(x):
            return -_full_score(x.reshape(-1, 2))
            
        try:
            res_full = minimize(
                _neg_full_score,
                points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if -res_full.fun > best_score:
                # Verify
                new_pts = res_full.x.reshape(-1, 2)
                if _full_score(new_pts) > best_score:
                    points = new_pts
                    best_score = _full_score(points)
        except Exception:
            pass

    return points
