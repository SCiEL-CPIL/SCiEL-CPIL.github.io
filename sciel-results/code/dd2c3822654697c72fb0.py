# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 570.0  # 60s safety margin before 630s kill

N_POINTS = 13
# Precompute triangle indices for n=13
TRIANGLES = np.array(list(itertools.combinations(range(N_POINTS), 3)))

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    """
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

def _soft_min_objective(params, temperature=0.01):
    """
    Smooth approximation of -A_min / A_hull using log-sum-exp for the min area.
    This makes the objective differentiable, allowing gradient-based methods like L-BFGS-B.
    """
    points = params.reshape(N_POINTS, 2)
    
    if not np.all(np.isfinite(points)):
        return 1e10
        
    p1 = points[TRIANGLES[:, 0]]
    p2 = points[TRIANGLES[:, 1]]
    p3 = points[TRIANGLES[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -1/t * log(sum(exp(-t * areas)))
    # As t -> inf, this approaches -min(areas)
    t = 1.0 / temperature
    max_neg_area = -np.min(areas) # For numerical stability in log-sum-exp
    log_sum_exp = max_neg_area + np.log(np.sum(np.exp(-t * areas - max_neg_area)))
    soft_min_area = -log_sum_exp / t
    
    a_hull = _hull_area(points)
    if a_hull < 1e-10:
        return 1e10
        
    return -soft_min_area / a_hull

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _concentric_rings(radii, angles_per_ring):
    """
    Generate points from concentric rings.
    radii: list of radii
    angles_per_ring: list of lists of angles
    """
    points = []
    for i, r in enumerate(radii):
        for theta in angles_per_ring[i]:
            points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points)

def _polish(points, maxiter=200, maxfev=2000):
    """
    Final polish using Powell's method on the exact (nonsmooth) objective.
    """
    if time.time() > _DEADLINE - 10:
        return points
        
    def objective(params):
        p = params.reshape(N_POINTS, 2)
        if not np.all(np.isfinite(p)):
            return 1e10
        return -_score(p)
        
    try:
        res = minimize(
            objective,
            points.flatten(),
            method='Powell',
            options={'maxiter': maxiter, 'maxfev': maxfev, 'ftol': 1e-12, 'xtol': 1e-12}
        )
        if res.fun < -_score(points):
            return res.x.reshape(N_POINTS, 2)
    except Exception:
        pass
    return points

def _optimize_topology(topology, initial_params, maxiter=500, maxfev=10000):
    """
    Optimize a specific concentric topology using L-BFGS-B on soft-min objective,
    then polish with Powell.
    """
    if time.time() > _DEADLINE - 30:
        return None, 0.0
        
    def concentric_objective(params):
        idx = 0
        radii = []
        angles = []
        for count in topology:
            r = abs(params[idx])
            idx += 1
            radii.append(r)
            ring_angles = []
            for _ in range(count):
                theta = params[idx]
                idx += 1
                ring_angles.append(theta)
            angles.append(ring_angles)
        
        points = _concentric_rings(radii, angles)
        
        # Penalize coincident points
        min_dist = np.inf
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                d = np.linalg.norm(points[i] - points[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-6:
            return 1e10 - min_dist
            
        return _soft_min_objective(points.flatten(), temperature=0.05)
        
    try:
        # Phase 1: L-BFGS-B with soft-min
        res_lbfgs = minimize(
            concentric_objective,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxfev, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        # Reconstruct points
        idx = 0
        radii = []
        angles = []
        for count in topology:
            r = abs(res_lbfgs.x[idx])
            idx += 1
            radii.append(r)
            ring_angles = []
            for _ in range(count):
                theta = res_lbfgs.x[idx]
                idx += 1
                ring_angles.append(theta)
            angles.append(ring_angles)
        
        points = _concentric_rings(radii, angles)
        
        # Phase 2: Powell polish
        points = _polish(points)
        
        score = _score(points)
        return points, score
        
    except Exception:
        return None, 0.0

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area ratio.
    """
    best_points = None
    best_score = -np.inf
    
    # --- Strategy 1: Regular Polygon Baseline ---
    points_reg = _regular_polygon(N_POINTS)
    best_points = points_reg
    best_score = _score(points_reg)
    
    # --- Strategy 2: Concentric Ring Topologies ---
    # Focus on topologies known to be strong for Heilbronn.
    # 7 outer + 6 inner is a common structure.
    
    topologies = [
        [1, 6, 6],   # 1 center, 6 inner, 6 outer
        [1, 5, 7],   # 1 center, 5 inner, 7 outer
        [1, 7, 5],   # 1 center, 7 inner, 5 outer
        [0, 6, 7],   # 6 inner, 7 outer
        [0, 7, 6],   # 7 inner, 6 outer
        [1, 12],     # 1 center, 12 outer
        [2, 5, 6],   # 2 inner, 5 mid, 6 outer
        [2, 6, 5],   # 2 inner, 6 mid, 5 outer
    ]
    
    rng = np.random.default_rng(12345)
    
    for topology in topologies:
        if time.time() > _DEADLINE - 30:
            break
            
        # Construct initial parameters
        # Format: [r1, theta1_1..theta1_k1, r2, theta2_1..theta2_k2, ...]
        init_params = []
        
        num_rings = len(topology)
        # Heuristic initial radii
        if topology[0] == 1:
            base_radii = [0.0] + list(np.linspace(0.4, 1.0, num_rings - 1))
        else:
            base_radii = list(np.linspace(0.3, 1.0, num_rings))
            
        for i, count in enumerate(topology):
            r_init = base_radii[i]
            init_params.append(r_init)
            
            # Angles: evenly spaced + noise
            thetas = np.linspace(0, 2*np.pi, count, endpoint=False) + rng.uniform(-0.2, 0.2, count)
            init_params.extend(thetas)
        
        init_params = np.array(init_params)
        
        # Optimize this topology
        pts, score = _optimize_topology(topology, init_params)
        
        if pts is not None and score > best_score:
            best_score = score
            best_points = pts
    
    # --- Strategy 3: Random Restarts with Soft-Min + Polish ---
    # Try a few random starts to escape local optima.
    
    num_restarts = 8
    for i in range(num_restarts):
        if time.time() > _DEADLINE - 15:
            break
        
        # Random points in unit disk, but ensure some spread
        theta = rng.uniform(0, 2*np.pi, N_POINTS)
        r = np.sqrt(rng.uniform(0, 1, N_POINTS)) # Uniform in area
        pts_rand = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        
        # Phase 1: L-BFGS-B with soft-min
        try:
            res_lbfgs = minimize(
                lambda p: _soft_min_objective(p, temperature=0.05),
                pts_rand.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-10}
            )
            pts_lbfgs = res_lbfgs.x.reshape(N_POINTS, 2)
            
            # Phase 2: Powell polish
            pts_polished = _polish(pts_lbfgs, maxiter=100, maxfev=1000)
            
            score = _score(pts_polished)
            if score > best_score:
                best_score = score
                best_points = pts_polished
        except Exception:
            pass

    # Final validation
    if best_points is None or best_points.shape != (N_POINTS, 2) or not np.all(np.isfinite(best_points)):
        best_points = points_reg
        
    return best_points

# EVOLVE-BLOCK-END
