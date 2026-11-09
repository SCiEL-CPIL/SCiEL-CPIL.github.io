# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_vertex_score(vertices):
    """
    Compute the per-vertex score for a single vertex of a polygon.
    vertices: (n, 2) array of polygon vertices in cyclic order.
    Returns the score for vertex 0.
    """
    n = len(vertices)
    diffs = vertices - vertices[0]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    other_dists = dists[1:]
    d = np.sort(other_dists)
    
    # Check for close points
    if d[0] < 0.05:
        return 0.0
        
    gaps = d[1:] - d[:-1]
    k = np.argmin(gaps)
    D = (d[k] + d[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    if k == 0:
        sel = d[0:4]
    elif k == 17:
        sel = d[15:19]
    else:
        sel = d[k-1:k+3]
        
    # Avoid division by zero or invalid values
    sel = np.maximum(sel, 1e-9)
    raw = np.mean(np.maximum(sel / D, D / sel))
    
    if raw < 1e-9:
        return 0.0
        
    score = min(1.0, 1.0 / raw)
    return score

def _polygon_min_score(polygon):
    """
    Compute the minimum vertex score over all vertices of a polygon.
    polygon: (n, 2) array.
    """
    n = len(polygon)
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(polygon)):
        return 0.0
        
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(centered ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        return 0.0
        
    normalized = centered / mean_dist
    
    # Check convexity
    # Compute edge vectors
    edges = np.diff(np.vstack([normalized, normalized[:1]]), axis=0)
    
    # Check for zero length edges (repeated points)
    edge_lens = np.linalg.norm(edges, axis=1)
    if np.any(edge_lens < 1e-10):
        return 0.0
        
    # Cross products to check turn direction
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    
    # All cross products must be non-zero and same sign
    if np.any(np.abs(cross) < 1e-10):
        return 0.0
        
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return 0.0
        
    # Total turning angle must be +-2*pi
    # Compute angles of edges
    edge_angles = np.arctan2(edges[:, 1], edges[:, 0])
    # Differences in angles
    d_angles = np.diff(np.vstack([edge_angles, edge_angles[:1] + 2*np.pi*signs[0]]), axis=0)
    # Normalize to (-pi, pi]
    d_angles = (d_angles + np.pi) % (2 * np.pi) - np.pi
    
    total_turn = np.sum(d_angles)
    
    if abs(abs(total_turn) - 2 * np.pi) > 1e-6:
        return 0.0
        
    min_score = 1.0
    for i in range(n):
        rotated = np.roll(normalized, -i, axis=0)
        v_score = _compute_vertex_score(rotated)
        if v_score < min_score:
            min_score = v_score
            
    return min_score

def _evaluate_radii(radii, angles):
    """
    Convert radii and angles to polygon and compute score.
    """
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    poly = np.stack([x, y], axis=1)
    return _polygon_min_score(poly)

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin
    
    n = 20
    # Fixed angles for regular polygon layout
    regular_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Initial best: regular polygon
    best_radii = np.ones(n)
    best_poly = np.stack([np.cos(regular_angles), np.sin(regular_angles)], axis=1)
    best_score = _polygon_min_score(best_poly)
    
    # Try a few simple radial perturbations manually first to get a better start
    # e.g., alternating 1, 1.1, 1, 1.1...
    for scale in [1.0, 1.05, 0.95, 1.1, 0.9]:
        r_test = np.ones(n) * scale
        s = _evaluate_radii(r_test, regular_angles)
        if s > best_score:
            best_score = s
            best_radii = r_test
            best_poly = np.stack([r_test * np.cos(regular_angles), r_test * np.sin(regular_angles)], axis=1)
            
    # Differential Evolution
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        return best_poly

    # Bounds for radii: [0.5, 2.0] relative to unit circle
    # This ensures convexity is likely maintained for small deviations,
    # but we rely on the score function to reject non-convex shapes.
    bounds = [(0.5, 2.0)] * n
    
    def objective(radii):
        return -_evaluate_radii(radii, regular_angles)

    # Run DE
    # popsize=15 for a reasonable balance of exploration and speed
    # maxiter=50 to keep within time limits
    # tol=1e-6 for convergence
    
    # We can run multiple short DE runs if time permits, or one long one.
    # Let's try one run with moderate settings first.
    
    # To increase the chance of finding a global optimum, we might want to
    # restart with different initial populations if time allows.
    
    n_restarts = 3
    for i in range(n_restarts):
        if time.time() > deadline - 60:
            break
            
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                seed=42 + i, 
                maxiter=50, 
                popsize=15, 
                tol=1e-6,
                init='latinhypercube'
            )
            
            if res.fun < -best_score:
                new_score = -res.fun
                if new_score > best_score:
                    best_score = new_score
                    best_radii = res.x
                    best_poly = np.stack([best_radii * np.cos(regular_angles), best_radii * np.sin(regular_angles)], axis=1)
                    
        except Exception:
            continue
            
    return best_poly

# EVOLVE-BLOCK-END
