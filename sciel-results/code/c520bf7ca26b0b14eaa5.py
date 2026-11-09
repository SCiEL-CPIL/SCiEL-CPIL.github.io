# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 580.0  # 50s safety margin

N_POINTS = 13
# Precompute triangle indices for n=13
TRIANGLES = np.array(list(itertools.combinations(range(N_POINTS), 3)))

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    """
    # points shape: (13, 2)
    p1 = points[TRIANGLES[:, 0]]
    p2 = points[TRIANGLES[:, 1]]
    p3 = points[TRIANGLES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    """
    Compute the area of the convex hull.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """
    Calculate the ratio A_min / A_hull.
    """
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    
    if a_hull < 1e-10:
        return 0.0
        
    return a_min / a_hull

def _objective_free(params):
    """
    Objective function for free-form optimization: minimize -score.
    """
    points = params.reshape(N_POINTS, 2)
    if not np.all(np.isfinite(points)):
        return 1e10
    s = _score(points)
    return -s

def _generate_concentric_rings(radii, angles, center=(0.0, 0.0)):
    """
    Generate points from a concentric ring parameterization.
    radii: list of radii for each ring
    angles: list of lists of angles for each ring
    """
    points = []
    # Add center if radius is 0 or if we define a center point separately
    # Here we assume the first ring can have radius 0 to represent the center
    
    for i, r in enumerate(radii):
        if r < 1e-9:
            # If radius is effectively 0, all angles map to the center
            # But we only want one point at the center if this is the center ring
            # To handle this cleanly, we usually treat the center as a separate point
            # or ensure only one point is assigned to r=0.
            # For this function, we just plot them. If multiple points map to (0,0), 
            # they are coincident, which is bad. 
            # So we assume the user provides valid distinct points or we handle deduplication later.
            pass
        
        for theta in angles[i]:
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            points.append([x, y])
            
    return np.array(points)

def _concentric_objective(params, topology):
    """
    Objective for concentric ring optimization.
    params: [r1, a1_1, a1_2, ..., r2, a2_1, ..., ...]
    topology: list of counts, e.g., [1, 6, 6]
    """
    idx = 0
    radii = []
    angles = []
    
    for count in topology:
        # Radius for this ring
        r = params[idx]
        idx += 1
        radii.append(abs(r)) # Ensure radius is positive
        
        # Angles for this ring
        ring_angles = []
        for _ in range(count):
            theta = params[idx]
            idx += 1
            ring_angles.append(theta)
        angles.append(ring_angles)
    
    points = _generate_concentric_rings(radii, angles)
    
    # Check for coincident points (bad for hull and min area)
    # If any two points are very close, penalize heavily
    if len(points) < 3:
        return 1e10
        
    # Simple check for duplicates
    # Using cdist would be fast, but let's just rely on the optimizer to avoid exact duplicates
    # or add a small penalty if min distance is tiny
    min_dist = np.inf
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-6:
        return 1e10 - min_dist # Penalize strongly
        
    s = _score(points)
    return -s

def _optimize_topology(topology, initial_params, maxiter=1000, maxfev=20000):
    """
    Optimize a specific concentric topology.
    """
    try:
        result = minimize(
            _concentric_objective,
            initial_params,
            args=(topology,),
            method='Powell',
            options={
                'maxiter': maxiter,
                'maxfev': maxfev,
                'ftol': 1e-10,
                'xtol': 1e-10
            }
        )
        
        # Reconstruct points to get the score
        idx = 0
        radii = []
        angles = []
        for count in topology:
            r = abs(result.x[idx])
            idx += 1
            radii.append(r)
            ring_angles = []
            for _ in range(count):
                theta = result.x[idx]
                idx += 1
                ring_angles.append(theta)
            angles.append(ring_angles)
        
        points = _generate_concentric_rings(radii, angles)
        score = _score(points)
        return points, score, result.fun
        
    except Exception as e:
        return None, 0.0, 1e10

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area ratio.
    """
    best_points = None
    best_score = -np.inf
    
    # --- Strategy 1: Free-form optimization from Regular Polygon ---
    # This is a strong baseline.
    angles_reg = np.linspace(0, 2 * np.pi, N_POINTS, endpoint=False)
    points_reg = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    
    x0_reg = points_reg.flatten()
    
    if time.time() < _DEADLINE - 100:
        try:
            res_reg = minimize(
                _objective_free,
                x0_reg,
                method='Powell',
                options={'maxiter': 2000, 'maxfev': 50000, 'ftol': 1e-12, 'xtol': 1e-12}
            )
            if res_reg.fun < -best_score:
                best_score = -res_reg.fun
                best_points = res_reg.x.reshape(N_POINTS, 2)
        except Exception:
            pass

    # --- Strategy 2: Concentric Ring Topologies ---
    # Try various distributions of points on concentric circles.
    # Common optimal structures for Heilbronn involve concentric rings.
    
    topologies = [
        [1, 6, 6],   # 1 center, 6 inner, 6 outer
        [1, 5, 7],   # 1 center, 5 inner, 7 outer
        [0, 6, 7],   # 6 inner, 7 outer (no center)
        [1, 12],     # 1 center, 12 outer
        [0, 13],     # 13 on one circle (regular polygon case, but optimized angles/radius)
        [1, 4, 8],   # 1 center, 4 inner, 8 outer
        [2, 5, 6],   # 2 inner (close to center), 5 mid, 6 outer
    ]
    
    rng = np.random.default_rng(42)
    
    for topology in topologies:
        if time.time() > _DEADLINE - 20:
            break
            
        # Construct initial parameters
        # Format: [r1, theta1_1..theta1_k1, r2, theta2_1..theta2_k2, ...]
        init_params = []
        
        # Heuristic initial radii: evenly spaced or based on sqrt(area)
        # For k rings, radii roughly proportional to sqrt(i/k)
        num_rings = len(topology)
        base_radii = np.sqrt(np.linspace(0.2, 1.0, num_rings))
        
        for i, count in enumerate(topology):
            r_init = base_radii[i] if i > 0 else 0.0 # First ring often center
            init_params.append(r_init)
            
            # Angles: evenly spaced + noise
            thetas = np.linspace(0, 2*np.pi, count, endpoint=False) + rng.uniform(-0.1, 0.1, count)
            init_params.extend(thetas)
        
        init_params = np.array(init_params)
        
        # Optimize this topology
        pts, score, fun_val = _optimize_topology(topology, init_params)
        
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts
    
    # --- Strategy 3: Random Restarts with Local Refinement ---
    # If we have time, try a few random starts and polish them.
    # This helps if the global optimum is not concentric.
    
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > _DEADLINE - 10:
            break
        
        # Random points in unit circle
        theta = rng.uniform(0, 2*np.pi, N_POINTS)
        r = rng.uniform(0, 1, N_POINTS)
        # Ensure no point is too close to center to avoid degeneracy? 
        # Actually, center is allowed.
        pts_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        
        x0 = pts_rand.flatten()
        
        try:
            res = minimize(
                _objective_free,
                x0,
                method='Powell',
                options={'maxiter': 500, 'maxfev': 10000, 'ftol': 1e-10, 'xtol': 1e-10}
            )
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape(N_POINTS, 2)
        except Exception:
            pass

    # Fallback
    if best_points is None or best_points.shape != (N_POINTS, 2):
        best_points = points_reg
        
    # Final validation
    if not np.all(np.isfinite(best_points)):
        best_points = points_reg
        
    return best_points

# EVOLVE-BLOCK-END
