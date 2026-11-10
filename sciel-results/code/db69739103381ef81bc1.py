# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_DEADLINE = _START_TIME + 580.0  # 20s safety margin

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Generate all combinations of 3 indices
    indices = np.array(list(itertools.combinations(range(n), 3)))
    
    # Extract points
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Calculate areas: 0.5 * | (p2-p1) x (p3-p1) |
    v1 = p2 - p1
    v2 = p3 - p1
    # Cross product in 2D is a scalar: v1_x * v2_y - v1_y * v2_x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the metric: A_min / A_hull.
    """
    a_hull = _hull_area(points)
    if a_hull <= 1e-12:
        return 0.0
    a_min = _min_triangle_area(points)
    return a_min / a_hull

def _soft_min_area(points, beta=50.0):
    """
    Compute a differentiable approximation of the minimum triangle area.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    """
    n = len(points)
    indices = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Numerical stability: shift by min
    min_a = np.min(areas)
    if min_a < 0: min_a = 0
    
    shifted = areas - min_a
    shifted = np.clip(shifted, 0, None)
    
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return min_a
    
    log_sum = np.log(sum_exp)
    soft_min = min_a - (1.0 / beta) * log_sum
    
    return soft_min

def _generate_concentric_rings(params):
    """
    Generate points based on concentric ring parameters.
    params: [r1, r2, r3, theta_offset1, theta_offset2, theta_offset3]
    We use 3 rings: 
    - Ring 1 (inner): 1 point (center) - actually let's do 3 rings of points.
    - To get 13 points, a good structure is 1 center + 5 inner + 7 outer? 
    - Or 3 + 4 + 6?
    - Or 1 + 5 + 7?
    - Let's try 1 center, 5 in ring 1, 7 in ring 2.
    - Or 3 in ring 1, 4 in ring 2, 6 in ring 3.
    
    Let's stick to a flexible ring structure defined by:
    - n_rings
    - points_per_ring
    - radii
    - angular shifts
    
    For simplicity and robustness, let's use a specific topology:
    Center (1) + Ring 1 (5) + Ring 2 (7) = 13 points.
    This is a common heuristic for maximizing min distance/area.
    
    params: [r1, r2, shift1, shift2]
    r1: radius of ring 1
    r2: radius of ring 2
    shift1: angular offset for ring 1
    shift2: angular offset for ring 2
    """
    r1, r2, shift1, shift2 = params
    
    # Ensure radii are positive and ordered
    if r1 < 0.1: r1 = 0.1
    if r2 < r1 * 1.1: r2 = r1 * 1.1
    
    points = []
    
    # Center
    points.append([0.0, 0.0])
    
    # Ring 1: 5 points
    angles1 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + shift1
    for a in angles1:
        points.append([r1 * np.cos(a), r1 * np.sin(a)])
        
    # Ring 2: 7 points
    angles2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + shift2
    for a in angles2:
        points.append([r2 * np.cos(a), r2 * np.sin(a)])
        
    return np.array(points)

def _objective_structured(params):
    """
    Objective function for the structured optimization.
    We maximize the soft-min area ratio.
    Since the ratio is scale-invariant, we can fix the scale by normalizing 
    the outer radius to 1.0 inside the generator or objective.
    Here, we normalize r2 to 1.0 effectively by scaling the final points,
    or we just optimize r1/r2 ratio and absolute scale doesn't matter for ratio.
    However, to be safe, we normalize the points to unit hull area before scoring?
    No, the score is A_min / A_hull.
    If we scale points by k, A_min scales by k^2, A_hull scales by k^2. Ratio is invariant.
    So we can just return -soft_min_area(points).
    But wait, if the points are very small, numerical issues might arise.
    Let's normalize the points to have max radius 1.0 before computing soft_min.
    """
    points = _generate_concentric_rings(params)
    
    # Normalize scale: divide by max radius
    max_r = np.max(np.linalg.norm(points, axis=1))
    if max_r < 1e-6:
        return 1e6
    points = points / max_r
    
    # Compute soft min area
    s = _soft_min_area(points, beta=100.0)
    
    # Penalize if points are too close (degeneracy)
    if s < 1e-6:
        return 1e6 - s
        
    return -s

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the x,y coordinates.
    """
    n = 13
    best_score = -1.0
    best_points = np.random.default_rng(0).random((n, 2))
    
    # Initial guess: 
    # r1 ~ 0.5, r2 ~ 1.0
    # Shifts to maximize separation
    
    # We will use differential evolution for global search over the 4 parameters,
    # followed by L-BFGS-B for local refinement.
    
    bounds = [
        (0.1, 0.9),   # r1
        (1.0, 1.5),   # r2 (relative to r1? No, absolute. Let's say r2 is fixed to 1.0 and r1 varies? 
                      # Actually, let's let both vary, but normalize inside.
                      # If we normalize inside, r2 is effectively 1.0 and r1 is r1/r2.
                      # So r1 is in (0, 1). r2 is redundant if normalized.
                      # Let's change params to [r1_ratio, shift1, shift2]
                      # r1_ratio = r1 / r2.
                      # r2 is fixed to 1.0 in the generation step after normalization?
                      # No, _generate_concentric_rings takes raw params.
                      # Let's redefine params: [r1, shift1, shift2] with r2 fixed to 1.0 in generator?
                      # Simpler: params [r1, shift1, shift2]. Generator sets r2=1.0.
                      # Then normalization is just for safety.
                      # Let's update _generate_concentric_rings to take r1, s1, s2 and set r2=1.0.
                      # Wait, the function _generate_concentric_rings currently takes 4 params.
                      # I will modify the call to use a lambda or just fix r2 in the bounds.
                      # Let's use bounds for r1 in (0.1, 0.9), r2 in (1.0, 1.0) effectively?
                      # No, let's keep 4 params but constrain r2 > r1.
                      # Differential evolution handles bounds.
                      # Let's set r1 in (0.1, 0.8), r2 in (1.0, 1.2).
                      # Shifts in (-pi, pi).
        (0.1, 0.8),   # r1
        (1.0, 1.2),   # r2
        (-np.pi, np.pi), # shift1
        (-np.pi, np.pi)  # shift2
    ]
    
    # Update _generate_concentric_rings to accept 5 params? 
    # No, current takes 4: r1, r2, shift1, shift2.
    # My bounds above have 5 items? No, I listed 5 tuples. 
    # r1, r2, shift1, shift2. That's 4 params.
    # Let's re-verify bounds list length.
    # bounds = [(0.1, 0.9), (1.0, 1.5), (-np.pi, np.pi), (-np.pi, np.pi)]
    # Yes, 4 bounds.
    
    # Corrected bounds:
    bounds = [
        (0.1, 0.9),   # r1
        (1.0, 1.5),   # r2
        (-np.pi, np.pi), # shift1
        (-np.pi, np.pi)  # shift2
    ]
    
    # Global Search: Differential Evolution
    try:
        de_res = differential_evolution(
            _objective_structured, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-4,
            popsize=15,
            polish=False, # We will polish manually
            updating='immediate',
            workers=1 # Ensure no multiprocessing issues
        )
        if time.time() > _DEADLINE - 30:
            # If DE takes too long, skip to best found so far? 
            # DE returns best. If it's still running, we can't interrupt easily in this simple script.
            # But DE is fast for 4D.
            pass
            
        best_params_de = de_res.x
        best_points_de = _generate_concentric_rings(best_params_de)
        # Normalize
        max_r = np.max(np.linalg.norm(best_points_de, axis=1))
        if max_r > 0:
            best_points_de = best_points_de / max_r
        
        best_score_de = _score(best_points_de)
        
        if best_score_de > best_score:
            best_score = best_score_de
            best_points = best_points_de.copy()
            
    except Exception as e:
        print(f"DE failed: {e}")
    
    # Local Refinement: L-BFGS-B from best DE result
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(
                _objective_structured, 
                best_params_de, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            best_params_lbfgs = res.x
            best_points_lbfgs = _generate_concentric_rings(best_params_lbfgs)
            max_r = np.max(np.linalg.norm(best_points_lbfgs, axis=1))
            if max_r > 0:
                best_points_lbfgs = best_points_lbfgs / max_r
            
            best_score_lbfgs = _score(best_points_lbfgs)
            
            if best_score_lbfgs > best_score:
                best_score = best_score_lbfgs
                best_points = best_points_lbfgs.copy()
                
        except Exception as e:
            print(f"L-BFGS-B failed: {e}")
            
    # Additional Strategy: Try a different topology if time permits
    # Topology 2: 3 rings: 3, 4, 6? 
    # Or 1 center, 6 ring 1, 6 ring 2 (current program's ring 2 strategy but optimized)
    # Let's try optimizing the "1-6-6" topology as well.
    
    if time.time() < _DEADLINE - 20:
        
        def _generate_1_6_6(params):
            r1, r2, shift1, shift2 = params
            if r1 < 0.1: r1 = 0.1
            if r2 < r1 * 1.1: r2 = r1 * 1.1
            
            points = []
            points.append([0.0, 0.0])
            
            angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + shift1
            for a in angles1:
                points.append([r1 * np.cos(a), r1 * np.sin(a)])
                
            angles2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + shift2
            for a in angles2:
                points.append([r2 * np.cos(a), r2 * np.sin(a)])
                
            return np.array(points)
        
        def _objective_1_6_6(params):
            points = _generate_1_6_6(params)
            max_r = np.max(np.linalg.norm(points, axis=1))
            if max_r < 1e-6:
                return 1e6
            points = points / max_r
            s = _soft_min_area(points, beta=100.0)
            if s < 1e-6:
                return 1e6 - s
            return -s
        
        bounds_1_6_6 = [
            (0.1, 0.9),   # r1
            (1.0, 1.5),   # r2
            (-np.pi, np.pi), # shift1
            (-np.pi, np.pi)  # shift2
        ]
        
        try:
            de_res2 = differential_evolution(
                _objective_1_6_6, 
                bounds_1_6_6, 
                seed=123, 
                maxiter=50, 
                tol=1e-4,
                popsize=10,
                polish=False,
                workers=1
            )
            best_params_de2 = de_res2.x
            best_points_de2 = _generate_1_6_6(best_params_de2)
            max_r = np.max(np.linalg.norm(best_points_de2, axis=1))
            if max_r > 0:
                best_points_de2 = best_points_de2 / max_r
            
            best_score_de2 = _score(best_points_de2)
            
            if best_score_de2 > best_score:
                best_score = best_score_de2
                best_points = best_points_de2.copy()
                
            # Polish
            if time.time() < _DEADLINE - 5:
                res2 = minimize(
                    _objective_1_6_6, 
                    best_params_de2, 
                    method='L-BFGS-B', 
                    bounds=bounds_1_6_6,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                best_params_lbfgs2 = res2.x
                best_points_lbfgs2 = _generate_1_6_6(best_params_lbfgs2)
                max_r = np.max(np.linalg.norm(best_points_lbfgs2, axis=1))
                if max_r > 0:
                    best_points_lbfgs2 = best_points_lbfgs2 / max_r
                
                best_score_lbfgs2 = _score(best_points_lbfgs2)
                
                if best_score_lbfgs2 > best_score:
                    best_score = best_score_lbfgs2
                    best_points = best_points_lbfgs2.copy()
                    
        except Exception as e:
            print(f"1-6-6 strategy failed: {e}")

    # Final check: Ensure valid shape and finite values
    if not np.all(np.isfinite(best_points)):
        # Fallback to a regular polygon
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
