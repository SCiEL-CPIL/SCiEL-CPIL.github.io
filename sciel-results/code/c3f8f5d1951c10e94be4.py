"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

# EVOLVE-BLOCK-START

def _triangle_areas(points):
    """
    Calculate areas of all triangles formed by triples of points.
    Returns an array of areas.
    """
    n = points.shape[0]
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    
    indices = np.array(indices)
    pi = points[indices[:, 0]]
    pj = points[indices[:, 1]]
    pk = points[indices[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        if np.any(np.isnan(points)):
            return 0.0
        
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
        
        min_area = np.min(_triangle_areas(points))
        return min_area / hull_area
    except Exception:
        return 0.0

def _score_from_angles(angles):
    """
    Calculate score given angles on the unit circle.
    angles: array of shape (N,)
    """
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return _score(points)

def _negative_score_from_angles(angles):
    """
    Objective function for minimization on the manifold of angles.
    """
    return -_score_from_angles(angles)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (limit 150s)
    
    best_score = -np.inf
    best_angles = None
    
    # Initial guess: Regular N-gon
    # Fix angle 0 to 0 to remove rotational symmetry
    regular_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # Strategy: Optimize the angles on the unit circle.
    # This is a 14-dimensional problem (or 13 if we fix one angle).
    # We will optimize all 14 angles but the symmetry will be broken by the optimizer.
    # To be safe and efficient, we fix angle[0] = 0 by only optimizing angles[1:]
    
    def _objective_constrained(x):
        # x is angles[1:]
        full_angles = np.concatenate(([0.0], x))
        # Sort angles? The set of points is what matters.
        # If angles cross each other, the "order" on the circle changes, 
        # but the set of points is the same. 
        # However, _triangle_areas uses the indices in the array passed.
        # The set of points {P(0), P(x0), ..., P(x12)} is the same regardless of order.
        # So we don't need to sort.
        return _negative_score_from_angles(full_angles)
    
    # Start from regular distribution (minus the first 0)
    init_x = regular_angles[1:]
    
    # Run Nelder-Mead which is robust for this non-smooth objective
    try:
        res = minimize(_objective_constrained, init_x, method='Nelder-Mead',
                       options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-10})
        current_score = -res.fun
        if current_score > best_score:
            best_score = current_score
            best_angles = np.concatenate(([0.0], res.x))
    except Exception:
        pass
    
    # Try a few more starts with small perturbations to ensure we didn't get stuck
    rng = np.random.default_rng(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        
        # Jitter the regular angles slightly
        jitter = rng.normal(0, 0.05, N)
        # Keep angle 0 fixed for the optimization, but we can jitter the others
        # Actually, let's just jitter all and let the optimizer handle it?
        # No, stick to the constrained manifold for speed.
        # We can rotate the whole configuration.
        rot = rng.uniform(0, 2*np.pi)
        angles = (regular_angles + jitter + rot) % (2*np.pi)
        
        # To use the constrained optimizer, we need to map this to x (angles[1:]-angles[0])?
        # It's easier to just use the full 14D space for these secondary starts if time permits,
        # but Nelder-Mead on 13D is fast.
        # Let's just use the full score function on the points for a quick local check?
        # No, let's stick to the angle manifold.
        
        # Construct a new starting point for the constrained optimization
        # Shift so min angle is 0? 
        sorted_angles = np.sort(angles)
        # Map to x: angles[1:] - angles[0] ? No, just take the relative angles.
        # It is simpler to just re-run the optimizer from a jittered regular start in the "relative" space.
        # Relative angles for regular: [2pi/14, 4pi/14, ...]
        # Jittered relative:
        rel_angles = np.diff(np.sort(angles))
        # Ensure sum is approx 2pi
        # The constrained function takes angles[1:] relative to angle[0]=0.
        # So we can just pass the jittered regular angles (shifted so first is 0)
        jittered_reg = regular_angles[1:] + rng.normal(0, 0.1, N-1)
        
        try:
            res2 = minimize(_objective_constrained, jittered_reg, method='Nelder-Mead',
                            options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
            current_score2 = -res2.fun
            if current_score2 > best_score:
                best_score = current_score2
                best_angles = np.concatenate(([0.0], res2.x))
        except Exception:
            pass

    # If optimization failed or score is bad, fallback to regular polygon
    if best_angles is None or best_score < 0.1:
        best_angles = regular_angles
    
    # Convert best angles to points
    # Note: The score is invariant to rotation and scaling.
    # The evaluator calculates min_area / hull_area.
    # Our score calculation does exactly that.
    
    final_points = np.column_stack([np.cos(best_angles), np.sin(best_angles)])
    
    return final_points
# EVOLVE-BLOCK-END
