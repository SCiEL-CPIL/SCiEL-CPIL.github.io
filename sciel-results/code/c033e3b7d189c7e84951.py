import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

# EVOLVE-BLOCK-START

def _triangle_area(p1, p2, p3):
    # 0.5 * | (p2-p1) x (p3-p1) |
    # In 2D, cross product is scalar: x1*y2 - y1*x2
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return 0.5 * abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))

def _get_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    # Ensure points are 2D array
    pts = np.asarray(points)
    if pts.shape != (N, 2):
        return 0.0
    
    # Calculate min triangle area
    min_area = np.inf
    # Precomputed indices for speed
    # There are 286 triangles
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                area = _triangle_area(pts[i], pts[j], pts[k])
                if area < min_area:
                    min_area = area
                    if min_area == 0.0:
                        return 0.0
    
    if not np.isfinite(min_area):
        return 0.0

    # Calculate convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        # Degenerate case
        return 0.0
    
    if hull_area <= 1e-12:
        return 0.0
        
    return min_area / hull_area

def _generate_symmetric_points(r1, r2, theta1, theta2):
    """
    Generate 13 points based on 1 center + 5 inner + 7 outer structure.
    """
    points = np.zeros((N, 2))
    
    # Center
    points[0] = [0.0, 0.0]
    
    # Inner Ring (5 points)
    for i in range(5):
        angle = theta1 + i * (2 * np.pi / 5)
        points[i+1] = [r1 * np.cos(angle), r1 * np.sin(angle)]
        
    # Outer Ring (7 points)
    for i in range(7):
        angle = theta2 + i * (2 * np.pi / 7)
        points[i+6] = [r2 * np.cos(angle), r2 * np.sin(angle)]
        
    return points

def _objective_symmetric(params):
    """
    Objective function for symmetric parameter optimization.
    Minimizes -score.
    """
    r1, r2, t1, t2 = params
    if r1 < 0 or r2 < 0:
        return 1.0 # Large penalty
    pts = _generate_symmetric_points(r1, r2, t1, t2)
    score = _get_score(pts)
    return -score

def _objective_general(points_flat):
    """
    Objective function for general coordinate optimization.
    Minimizes -score.
    """
    pts = points_flat.reshape((N, 2))
    score = _get_score(pts)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Pre-compute triangle indices for potential speedup if needed, 
    # but _get_score is already reasonably fast for N=13.
    
    # Strategy 1: Symmetric 1+5+7 Optimization
    # Initial guesses for r1, r2, t1, t2
    # r1, r2 are radii. t1, t2 are angles.
    
    # We can fix t1 to 0 due to rotational symmetry of the inner ring relative to center?
    # Actually, the relative angle between inner and outer rings matters.
    # Let's optimize r1, r2, delta_theta where t1=0, t2=delta_theta.
    
    # Grid search for initial points
    r1_vals = np.linspace(0.2, 0.8, 10)
    r2_vals = np.linspace(0.9, 1.5, 10)
    t2_vals = np.linspace(0, np.pi, 10) # Symmetry of pentagon/heptagon allows reducing range
    
    best_sym_params = None
    best_sym_score = -1.0
    
    # Coarse Grid Search
    for r1 in r1_vals:
        if time.time() > deadline: break
        for r2 in r2_vals:
            if time.time() > deadline: break
            for t2 in t2_vals:
                # t1 is fixed at 0
                pts = _generate_symmetric_points(r1, r2, 0.0, t2)
                score = _get_score(pts)
                if score > best_sym_score:
                    best_sym_score = score
                    best_sym_params = (r1, r2, 0.0, t2)
                
    if best_sym_params is not None:
        # Local Refinement of symmetric parameters
        try:
            res = minimize(_objective_symmetric, best_sym_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            if time.time() < deadline:
                best_sym_params_opt = res.x
                # Ensure radii are positive
                if best_sym_params_opt[0] > 0 and best_sym_params_opt[1] > 0:
                    pts_opt = _generate_symmetric_points(best_sym_params_opt[0], best_sym_params_opt[1], best_sym_params_opt[2], best_sym_params_opt[3])
                    score_opt = _get_score(pts_opt)
                    if score_opt > best_score:
                        best_score = score_opt
                        best_points = pts_opt
        except Exception:
            pass

    # Strategy 2: General Local Search from Best Symmetric Solution
    if best_points is not None and time.time() < deadline:
        # Flatten points
        x0 = best_points.flatten()
        
        # Use multiple restarts or just one strong local search
        # Nelder-Mead on 26 dimensions can be slow/unstable, COBYLA might be better for constrained? 
        # No constraints other than geometry, but we want to avoid degenerate hulls.
        # Let's try Powell method which is often robust for this.
        
        try:
            res_general = minimize(_objective_general, x0, method='Powell', 
                                   options={'maxiter': 200, 'ftol': 1e-8, 'xtol': 1e-8})
            if time.time() < deadline:
                pts_general = res_general.x.reshape((N, 2))
                score_general = _get_score(pts_general)
                if score_general > best_score:
                    best_score = score_general
                    best_points = pts_general
        except Exception:
            pass
            
        # If time permits, try another local search from a slightly perturbed version
        if time.time() < deadline - 10:
            # Perturb best points slightly
            rng = np.random.default_rng(42)
            perturbed = best_points + rng.normal(0, 0.05, size=(N, 2))
            x0_pert = perturbed.flatten()
            try:
                res_pert = minimize(_objective_general, x0_pert, method='Nelder-Mead', 
                                    options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6})
                if time.time() < deadline:
                    pts_pert = res_pert.x.reshape((N, 2))
                    score_pert = _get_score(pts_pert)
                    if score_pert > best_score:
                        best_score = score_pert
                        best_points = pts_pert
            except Exception:
                pass

    # Fallback: Random search if nothing else worked or as a final check
    if best_points is None or time.time() < deadline - 5:
        # A simple random search with local refinement
        rng = np.random.default_rng(123)
        for _ in range(5): # 5 random starts
            if time.time() > deadline - 5: break
            # Generate random points in a square
            pts_rand = rng.random((N, 2)) * 2 - 1 # [-1, 1]
            # Center them
            pts_rand -= np.mean(pts_rand, axis=0)
            
            # Quick local refine
            try:
                res_rand = minimize(_objective_general, pts_rand.flatten(), method='Nelder-Mead', 
                                    options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5})
                if time.time() < deadline - 5:
                    pts_refined = res_rand.x.reshape((N, 2))
                    score_refined = _get_score(pts_refined)
                    if score_refined > best_score:
                        best_score = score_refined
                        best_points = pts_refined
            except Exception:
                pass

    if best_points is None:
        # Absolute fallback: regular 13-gon? Or just the initial symmetric guess
        # Use the best symmetric guess if available, else random
        if best_sym_params is not None:
            best_points = _generate_symmetric_points(best_sym_params[0], best_sym_params[1], best_sym_params[2], best_sym_params[3])
        else:
            # Regular 13-gon
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            best_points = np.column_stack((np.cos(angles), np.sin(angles)))
            
    return best_points

# EVOLVE-BLOCK-END
