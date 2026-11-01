import numpy as np
import scipy.spatial as spat
import time
from scipy.optimize import minimize

N = 14
DEADLINE = 145  # Safety margin before 150s kill

def _calc_min_tri_area(points):
    """
    Compute the minimum triangle area over all triples.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Create index arrays for unique triples i < j < k
    # Using combinations is safer and cleaner for n=14 (364 triples)
    idxs = np.array(list(itertools.combinations(range(n), 3)))
    
    p_i = points[idxs[:, 0]]
    p_j = points[idxs[:, 1]]
    p_k = points[idxs[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score = min triangle area / hull area. Higher is better."""
    hull_a = _hull_area(points)
    if hull_a < 1e-10:
        return 0.0
    min_tri = _calc_min_tri_area(points)
    return min_tri / hull_a

def _generate_two_heptagons(rho, phi):
    """
    Generate 14 points: 7 on outer unit circle, 7 on inner circle of radius rho,
    rotated by phi relative to outer.
    """
    # Outer heptagon (indices 0-6)
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    x_outer = np.cos(angles_outer)
    y_outer = np.sin(angles_outer)
    
    # Inner heptagon (indices 7-13)
    angles_inner = angles_outer + phi
    x_inner = rho * np.cos(angles_inner)
    y_inner = rho * np.sin(angles_inner)
    
    xs = np.concatenate([x_outer, x_inner])
    ys = np.concatenate([y_outer, y_inner])
    
    return np.column_stack([xs, ys])

def _score_two_heptagons(params):
    """
    Objective function for Nelder-Mead on the 2-parameter space.
    Returns negative score (for minimization).
    """
    rho, phi = params
    if rho < 0.1 or rho > 0.95:
        return 1e6
    
    pts = _generate_two_heptagons(rho, phi)
    
    # Normalize to unit area
    hull_a = _hull_area(pts)
    if hull_a < 1e-10:
        return 1e6
    pts = pts / np.sqrt(hull_a)
    
    score = _score(pts)
    return -score

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = -np.inf
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Phase 1: Optimize the 2-parameter geometric model
    # Initial guess: rho=0.5, phi=pi/7
    x0 = np.array([0.5, np.pi / 7.0])
    
    # Define bounds for rho and phi
    # Nelder-Mead doesn't support bounds directly, so we use a penalty or just rely on the function
    # But we can wrap it or just be careful. Let's use a simple wrapper that penalizes out-of-bounds.
    
    def objective(x):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            return 1e10
        return _score_two_heptagons(x)
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'fatol': 1e-12, 'xatol': 1e-12, 'disp': False})
        
        if res.fun < 1e9:
            best_params = res.x
            best_pts = _generate_two_heptagons(best_params[0], best_params[1])
            hull_a = _hull_area(best_pts)
            if hull_a > 1e-10:
                best_pts = best_pts / np.sqrt(hull_a)
                best_s = _score(best_pts)
                update_best(best_pts, best_s)
                
    except Exception:
        pass
    
    # If Phase 1 failed or didn't find anything good, try a grid search
    if best_points is None or best_score < 40.0:
        for rho in np.linspace(0.3, 0.8, 20):
            for phi in np.linspace(0, np.pi, 20, endpoint=False):
                elapsed = time.time() - start_time
                if elapsed > DEADLINE - 20:
                    break
                pts = _generate_two_heptagons(rho, phi)
                hull_a = _hull_area(pts)
                if hull_a < 1e-10:
                    continue
                pts = pts / np.sqrt(hull_a)
                s = _score(pts)
                update_best(pts, s)
        
        if best_points is None:
            # Fallback to regular polygon
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            best_points = np.column_stack([np.cos(angles), np.sin(angles)])
            best_score = _score(best_points)

    # Phase 2: Local refinement using coordinate descent on full 28D space
    # This allows breaking the symmetry of the two-heptagon model if it leads to a better score.
    # We use a simple coordinate descent with line search.
    
    current_pts = best_points.copy()
    current_pts = current_pts / np.sqrt(_hull_area(current_pts))
    current_score = _score(current_pts)
    
    rng = np.random.RandomState(42)
    
    # Precompute indices for speed? No, _score is fast enough for 14 points.
    # We will iterate over each of the 28 coordinates.
    
    max_iter = 200
    for it in range(max_iter):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        improved = False
        for i in range(N):
            for coord in range(2):
                # Try moving point i in coordinate coord
                # Use a small step size
                step = 0.01
                
                # Try positive step
                new_pts = current_pts.copy()
                new_pts[i, coord] += step
                hull_a = _hull_area(new_pts)
                if hull_a > 1e-10:
                    new_pts = new_pts / np.sqrt(hull_a)
                    new_score = _score(new_pts)
                    if new_score > current_score:
                        current_pts = new_pts
                        current_score = new_score
                        improved = True
                        update_best(current_pts, current_score)
                        continue # Re-evaluate other coordinates with new point
                
                # Try negative step
                new_pts = current_pts.copy()
                new_pts[i, coord] -= step
                hull_a = _hull_area(new_pts)
                if hull_a > 1e-10:
                    new_pts = new_pts / np.sqrt(hull_a)
                    new_score = _score(new_pts)
                    if new_score > current_score:
                        current_pts = new_pts
                        current_score = new_score
                        improved = True
                        update_best(current_pts, current_score)
        
        if not improved:
            # If no improvement with step 0.01, try smaller step
            step = 0.001
            for i in range(N):
                for coord in range(2):
                    new_pts = current_pts.copy()
                    new_pts[i, coord] += step
                    hull_a = _hull_area(new_pts)
                    if hull_a > 1e-10:
                        new_pts = new_pts / np.sqrt(hull_a)
                        new_score = _score(new_pts)
                        if new_score > current_score:
                            current_pts = new_pts
                            current_score = new_score
                            improved = True
                            update_best(current_pts, current_score)
                            break
                if improved:
                    break
                    
            if not improved:
                step = 0.0001
                for i in range(N):
                    for coord in range(2):
                        new_pts = current_pts.copy()
                        new_pts[i, coord] += step
                        hull_a = _hull_area(new_pts)
                        if hull_a > 1e-10:
                            new_pts = new_pts / np.sqrt(hull_a)
                            new_score = _score(new_pts)
                            if new_score > current_score:
                                current_pts = new_pts
                                current_score = new_score
                                improved = True
                                update_best(current_pts, current_score)
                                break
                    if improved:
                        break
                if not improved:
                    break # Convergence reached

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return np.array(best_points)

# Need to import itertools
import itertools
