import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

# Precompute triple indices for vectorized area calculation
_i_list, _j_list, _k_list = [], [], []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _i_list.append(_i)
            _j_list.append(_j)
            _k_list.append(_k)
_TRIPLE_I = np.array(_i_list)
_TRIPLE_J = np.array(_j_list)
_TRIPLE_K = np.array(_k_list)
_N_TRIPLES = len(_i_list)

def _compute_areas(points):
    """Vectorized triangle areas for all triples."""
    p1 = points[_TRIPLE_I]
    p2 = points[_TRIPLE_J]
    p3 = points[_TRIPLE_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def get_score(points):
    """Score = min triangle area / convex hull area."""
    ha = _hull_area(points)
    if ha <= 1e-12:
        return 0.0
    areas = _compute_areas(points)
    ma = np.min(areas)
    if ma < 1e-15:
        return 0.0
    return ma / ha

def _score_neg(points):
    return -get_score(points)

def _regular_polygon(radius=1.0):
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y))

def _ring_config_to_points(radii, angles_per_ring):
    """
    radii: array of ring radii
    angles_per_ring: list of arrays of angles for each ring
    Returns (N, 2) array
    """
    points = []
    for r, angles in zip(radii, angles_per_ring):
        for a in angles:
            points.append([r * np.cos(a), r * np.sin(a)])
    return np.array(points)

def _optimize_ring_topology(n_rings, points_per_ring, rng):
    """
    Optimize a specific ring topology.
    points_per_ring: list of integers summing to N
    """
    # Parameters: radii (n_rings) + angles (N)
    # We fix the first angle of the first ring to 0 to break rotational symmetry.
    # So we optimize N-1 angles.
    # Total params: n_rings + N - 1
    
    def objective(params):
        radii = params[:n_rings]
        angles_opt = params[n_rings:]
        
        # Ensure radii are positive
        radii = np.abs(radii) + 1e-6
        
        # Reconstruct full angles
        # First ring: first angle is 0
        full_angles = np.zeros(N)
        idx = 0
        for i, n_pts in enumerate(points_per_ring):
            if i == 0:
                # First point is fixed at 0
                full_angles[idx] = 0.0
                idx += 1
                # Remaining points in this ring
                if n_pts > 1:
                    full_angles[idx:idx+n_pts-1] = angles_opt[:n_pts-1]
                    angles_opt = angles_opt[n_pts-1:]
                    idx += n_pts - 1
            else:
                full_angles[idx:idx+n_pts] = angles_opt[:n_pts]
                angles_opt = angles_opt[n_pts:]
                idx += n_pts
                
        angles_per_ring = []
        idx = 0
        for n_pts in points_per_ring:
            angles_per_ring.append(full_angles[idx:idx+n_pts])
            idx += n_pts
            
        points = _ring_config_to_points(radii, angles_per_ring)
        return _score_neg(points)
        
    # Initial guess
    init_radii = np.linspace(0.5, 1.0, n_rings)
    # Initial angles: evenly spaced per ring
    init_angles_opt = []
    for i, n_pts in enumerate(points_per_ring):
        if i == 0:
            # First point fixed at 0
            if n_pts > 1:
                # Spacing for remaining
                step = 2*np.pi / n_pts
                init_angles_opt.extend([k*step for k in range(1, n_pts)])
        else:
            # Offset by half step relative to previous? Just uniform for now
            step = 2*np.pi / n_pts
            init_angles_opt.extend([k*step for k in range(n_pts)])
            
    init_params = np.concatenate([init_radii, np.array(init_angles_opt)])
    
    # Add some noise to break symmetry
    init_params += rng.normal(0, 0.01, len(init_params))
    
    result = minimize(objective, init_params, method='Powell', 
                      options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-6})
    
    if result.success or result.fun < 0:
        best_params = result.x
        radii = np.abs(best_params[:n_rings]) + 1e-6
        angles_opt = best_params[n_rings:]
        
        # Reconstruct angles same as in objective
        full_angles = np.zeros(N)
        idx = 0
        for i, n_pts in enumerate(points_per_ring):
            if i == 0:
                full_angles[idx] = 0.0
                idx += 1
                if n_pts > 1:
                    full_angles[idx:idx+n_pts-1] = angles_opt[:n_pts-1]
                    angles_opt = angles_opt[n_pts-1:]
                    idx += n_pts - 1
            else:
                full_angles[idx:idx+n_pts] = angles_opt[:n_pts]
                angles_opt = angles_opt[n_pts:]
                idx += n_pts
                
        angles_per_ring = []
        idx = 0
        for n_pts in points_per_ring:
            angles_per_ring.append(full_angles[idx:idx+n_pts])
            idx += n_pts
            
        points = _ring_config_to_points(radii, angles_per_ring)
        return points, -result.fun
    else:
        return None, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    rng = np.random.default_rng(42)
    
    best_points = _regular_polygon(1.0)
    best_score = get_score(best_points)
    
    # List of topologies to try
    # (n_rings, points_per_ring)
    topologies = [
        (1, [14]),
        (2, [7, 7]),
        (2, [6, 8]),
        (2, [5, 9]),
        (2, [4, 10]),
        (3, [5, 5, 4]),
        (3, [4, 5, 5]),
        (3, [3, 5, 6]),
        (3, [4, 4, 6]),
        (4, [4, 4, 3, 3]),
        (4, [3, 4, 4, 3]),
    ]
    
    # Try each topology
    for n_rings, pts_per_ring in topologies:
        if time.time() > deadline - 10.0:
            break
            
        try:
            pts, score = _optimize_ring_topology(n_rings, pts_per_ring, rng)
            if pts is not None and score > best_score:
                best_score = score
                best_points = pts
        except Exception:
            pass
            
    # Final polish: General (x,y) optimization from the best ring solution
    if time.time() < deadline - 5.0:
        try:
            # Use the best points found so far as initial guess
            init_flat = best_points.flatten()
            
            def obj_flat(x):
                p = x.reshape(N, 2)
                return _score_neg(p)
                
            result = minimize(obj_flat, init_flat, method='Powell', 
                              options={'maxiter': 300, 'xtol': 1e-8, 'ftol': 1e-8})
            
            if result.success or result.fun < 0:
                pts = result.x.reshape(N, 2)
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points = pts
        except Exception:
            pass

    return best_points
