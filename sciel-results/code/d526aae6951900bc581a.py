# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

N = 300

def _build_points(radii, phases, ring_counts):
    points = []
    for i, (r, phase, count) in enumerate(zip(radii, phases, ring_counts)):
        if count == 0:
            continue
        if count == 1:
            # Single point at center or radius r
            # If r is 0, it's at origin. If r > 0, it's at (r, 0) rotated by phase.
            # Actually, for count=1, the angle is 0 + phase.
            angle = phase
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append(np.array([[x, y]]))
        else:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + phase
            xs = r * np.cos(angles)
            ys = r * np.sin(angles)
            points.append(np.column_stack([xs, ys]))
    
    if not points:
        return np.zeros((0, 2))
    
    return np.vstack(points)

def _compute_radius(points):
    if len(points) == 0:
        return 0.0
    if len(points) == 1:
        return 1.0 - np.hypot(*points[0])
    
    dists_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dists_boundary)
    
    dist_matrix = cdist(points, points)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix) / 2.0
    
    return min(min_boundary, min_pair)

def _objective(params, ring_counts):
    k = len(ring_counts)
    radii = params[:k]
    phases = params[k:]
    
    radii = np.maximum(radii, 0.0)
    
    points = _build_points(radii, phases, ring_counts)
    
    if len(points) != N:
        return 0.0
        
    r = _compute_radius(points)
    return -r

def _optimize_structure(ring_counts):
    num_rings = len(ring_counts)
    
    # Initial guess
    x0 = np.zeros(num_rings * 2)
    for i, n in enumerate(ring_counts):
        if n > 1:
            # Estimate radius based on circumference fitting
            # 2 * pi * r_i >= n * 2 * r_circle
            # Assume r_circle ~ 0.05
            r_est = (n * 0.05) / np.pi
            x0[i] = r_est
        else:
            x0[i] = 0.0
        x0[num_rings + i] = 0.0 # Phase

    # Bounds
    bounds = [(0.0, 1.0)] * num_rings + [(0.0, 2 * np.pi)] * num_rings
    
    # Global Search
    # DE can be slow, but robust. 
    # We use a small population and limited iterations to fit in time.
    try:
        de_result = differential_evolution(
            _objective, 
            bounds, 
            args=(ring_counts,),
            maxiter=100, 
            popsize=15, 
            tol=1e-5,
            seed=42,
            init='latinhypercube',
            polish=True # Use local optimization at the end
        )
        best_x = de_result.x
        best_val = de_result.fun
    except Exception:
        best_x = x0
        best_val = _objective(x0, ring_counts)
    
    # Local Refinement
    # Powell is good for non-smooth functions
    local_result = minimize(
        _objective, 
        best_x, 
        args=(ring_counts,),
        method='Powell',
        options={'maxiter': 2000, 'xtol': 1e-8, 'ftol': 1e-10}
    )
    
    if local_result.fun < best_val:
        best_x = local_result.x
        best_val = local_result.fun
    
    radii = np.maximum(best_x[:num_rings], 0.0)
    phases = best_x[num_rings:]
    
    points = _build_points(radii, phases, ring_counts)
    
    if len(points) != N:
        return None, 0.0
        
    r = _compute_radius(points)
    
    return points, r

def construct():
    # Try a few different ring structures
    structures = [
        np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29]),
        np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 50, 33]), # Adjusted last ring
        np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29])  # Duplicate for safety, maybe change counts
    ]
    
    # Let's generate a more balanced structure
    # Sum must be 300.
    # Try: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29 is 300.
    # Try: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29
    
    best_points = None
    best_r = 0.0
    
    for struct in structures:
        if np.sum(struct) != N:
            continue
        pts, r = _optimize_structure(struct)
        if pts is not None and r > best_r:
            best_r = r
            best_points = pts
    
    if best_points is None:
        # Fallback
        return np.zeros((N, 2))
        
    return best_points
# EVOLVE-BLOCK-END
