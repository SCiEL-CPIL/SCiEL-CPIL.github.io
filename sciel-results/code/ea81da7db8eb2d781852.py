# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_vertex_score(vertices):
    """
    Compute the per-vertex score for a single vertex of a polygon.
    vertices: (n, 2) array of polygon vertices in cyclic order.
    Returns the score for vertex 0 (we rotate so the target vertex is index 0).
    """
    n = len(vertices)
    # Distances from vertex 0 to all others
    diffs = vertices - vertices[0]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    # Exclude self
    other_dists = dists[1:]
    # Sort
    d = np.sort(other_dists)
    # Check min distance
    if d[0] < 0.05:
        return 0.0
    # Compute gaps
    gaps = d[1:] - d[:-1]
    # Find first index minimizing the gap d_{k+1} - d_k
    # k is index in d, so we look at gaps[k] = d[k+1] - d[k]
    k = np.argmin(gaps)
    D = (d[k] + d[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    # Select four distances
    if k == 0:
        sel = d[0:4]
    elif k == 17:
        sel = d[15:19]
    else:
        sel = d[k-1:k+3]
    raw = np.mean(np.maximum(sel / D, D / sel))
    score = min(1.0, 1.0 / raw)
    return score

def _polygon_min_score(polygon):
    """
    Compute the minimum vertex score over all vertices of a polygon.
    polygon: (n, 2) array.
    """
    n = len(polygon)
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(centered ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    normalized = centered / mean_dist
    
    # Check convexity
    # No repeated points
    if np.any(np.linalg.norm(np.diff(np.vstack([normalized, normalized[:1]]), axis=0), axis=1) < 1e-10):
        return 0.0
    # Compute cross products for turns
    edges = np.diff(np.vstack([normalized, normalized[:1]]), axis=0)
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    # Check all turns are same sign and nonzero
    if np.any(np.abs(cross) < 1e-10):
        return 0.0
    # Check all same sign
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return 0.0
    # Total turning should be +-2*pi
    total_turn = np.sum(np.arctan2(cross, edges[:, 0] * edges[:, 0] + edges[:, 1] * edges[:, 1]))
    if abs(abs(total_turn) - 2 * np.pi) > 1e-6:
        return 0.0
    
    # Compute min vertex score
    min_score = 1.0
    for i in range(n):
        # Rotate so vertex i is at index 0
        rotated = np.roll(normalized, -i, axis=0)
        v_score = _compute_vertex_score(rotated)
        if v_score < min_score:
            min_score = v_score
    return min_score

def _make_candidate_rbf(params, n=20):
    """
    Create a polygon using Radial Basis Function parameterization.
    params: array of shape (2*M,) where M is the number of harmonics.
    params[0::2] are amplitudes for cos, params[1::2] are amplitudes for sin.
    r_i = 1 + sum(A_j * cos(j * theta_i) + B_j * sin(j * theta_i))
    """
    M = len(params) // 2
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = np.ones(n)
    for j in range(1, M + 1):
        A = params[2*(j-1)]
        B = params[2*(j-1)+1]
        radii += A * np.cos(j * angles) + B * np.sin(j * angles)
    
    # Ensure convexity by enforcing a minimum radius
    # The derivative of r(theta) = 1 + sum... must satisfy |dr/dtheta| < r
    # For small amplitudes, this is guaranteed. We clamp to be safe.
    radii = np.maximum(radii, 0.5)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _objective(params):
    """
    Objective function for minimization (negative of the score).
    """
    try:
        polygon = _make_candidate_rbf(params)
        score = _polygon_min_score(polygon)
        return -score
    except Exception:
        return 1000.0

def _nelder_mead(func, x0, maxiter=1000, tol=1e-6):
    """
    Simple implementation of Nelder-Mead algorithm.
    """
    n = len(x0)
    # Initialize simplex
    simplex = [x0]
    for i in range(n):
        x_new = x0.copy()
        x_new[i] += 0.1 if x_new[i] == 0 else 0.1 * x_new[i]
        simplex.append(x_new)
    
    f_values = [func(x) for x in simplex]
    
    for iteration in range(maxiter):
        # Sort simplex by function value
        order = np.argsort(f_values)
        simplex = [simplex[i] for i in order]
        f_values = [f_values[i] for i in order]
        
        # Check convergence
        if abs(f_values[0] - f_values[-1]) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = func(x_r)
        
        if f_r < f_values[0]:
            # Expansion
            gamma = 2.0
            x_e = centroid + gamma * (centroid - simplex[-1])
            f_e = func(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_values[-1] = f_e
            else:
                simplex[-1] = x_r
                f_values[-1] = f_r
        elif f_r < f_values[-2]:
            # Accept reflection
            simplex[-1] = x_r
            f_values[-1] = f_r
        else:
            # Contraction
            rho = 0.5
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = func(x_c)
            if f_c < f_values[-1]:
                simplex[-1] = x_c
                f_values[-1] = f_c
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    f_values[i] = func(simplex[i])
                    
    return simplex[0], f_values[0]

def _search():
    """
    Search for a good 20-gon using RBF parameterization and Nelder-Mead.
    """
    n = 20
    start_time = time.time()
    deadline = start_time + 620.0  # 10s safety margin
    
    best_score = 0.0
    best_polygon = None
    
    # Start with regular polygon (all params 0)
    num_harmonics = 4
    num_params = 2 * num_harmonics
    x0 = np.zeros(num_params)
    
    # Try a few random starting points to avoid local minima
    np.random.seed(42)
    
    # 1. Start from regular
    x_opt, f_opt = _nelder_mead(_objective, x0, maxiter=200)
    poly = _make_candidate_rbf(x_opt)
    score = _polygon_min_score(poly)
    best_score = score
    best_polygon = poly
    
    # 2. Try some random perturbations
    for i in range(10):
        if time.time() > deadline:
            break
        
        x_rand = np.random.normal(0, 0.1, num_params)
        x_opt, f_opt = _nelder_mead(_objective, x_rand, maxiter=100)
        poly = _make_candidate_rbf(x_opt)
        score = _polygon_min_score(poly)
        
        if score > best_score:
            best_score = score
            best_polygon = poly
            
    return best_polygon

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    
    # Try search-based approach
    polygon = _search()
    
    # Ensure we don't exceed time limit
    if time.time() - start_time > 625:
        # Fallback to regular polygon
        n = 20
        radius = 1.0
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        polygon = np.stack([x, y], axis=1)
    
    return polygon

# EVOLVE-BLOCK-END
