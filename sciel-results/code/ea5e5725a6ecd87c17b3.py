# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time

def _vertex_score(dists):
    """
    Compute the vertex score given the 19 distances from a vertex to the other 19 vertices.
    dists: array of shape (19,) with sorted distances d_0 <= ... <= d_18
    """
    # Check for very small distances
    if dists[0] < 0.05:
        return 0.0
    
    # Find k: first index minimizing d_{k+1} - d_k
    gaps = np.diff(dists)  # gaps[i] = d_{i+1} - d_i, length 18
    k = int(np.argmin(gaps))
    D = (dists[k] + dists[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
    
    # Select four distances
    if k == 0:
        selected = dists[0:4]
    elif k == 17:
        selected = dists[15:19]
    else:
        selected = dists[k-1:k+3]
    
    # raw = mean of max(d/D, D/d) for the four selected distances
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    
    if raw <= 0:
        return 0.0
    
    return min(1.0, 1.0 / raw)


def _compute_min_vertex_score(polygon):
    """
    Compute the minimum vertex score for a given polygon (20, 2).
    Returns 0 if invalid, otherwise the min vertex score.
    """
    n = len(polygon)
    if n != 20:
        return 0.0
    
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(centered**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        return 0.0
    
    normalized = centered / mean_dist
    
    # Check for repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(normalized[i] - normalized[j]) < 1e-10:
                return 0.0
    
    # Check convexity: every turn between consecutive edges nonzero and of one sign
    # Compute cross products of consecutive edge vectors
    cross_signs = []
    for i in range(n):
        p0 = normalized[i]
        p1 = normalized[(i+1) % n]
        p2 = normalized[(i+2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross) < 1e-12:
            return 0.0
        cross_signs.append(np.sign(cross))
    
    # All cross products must have the same sign
    if len(set(cross_signs)) != 1:
        return 0.0
    
    # Compute total turning angle; should be +-2*pi
    # For a strictly convex polygon, total turning is 2*pi (or -2*pi)
    # We can verify by checking that the sum of signed angles is approximately 2*pi
    total_turn = 0.0
    for i in range(n):
        p0 = normalized[i]
        p1 = normalized[(i+1) % n]
        p2 = normalized[(i+2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        # Signed angle from v1 to v2
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        dot = np.dot(v1, v2)
        angle = np.arctan2(cross, dot)
        total_turn += angle
    
    if abs(abs(total_turn) - 2*np.pi) > 1e-6:
        return 0.0
    
    # Compute vertex scores
    min_score = 1.0
    for i in range(n):
        vi = normalized[i]
        # Distances to all other vertices
        d = np.sqrt(np.sum((normalized - vi)**2, axis=1))
        # Remove distance to self
        d = np.delete(d, i)
        d = np.sort(d)
        
        score = _vertex_score(d)
        min_score = min(min_score, score)
    
    return min_score


def _fitness(params):
    """
    Fitness function for differential_evolution.
    params: array of 20 angles (one per vertex), which we map to points on a circle
            with possibly varying radii. Actually, let's parameterize more flexibly.
    
    We'll use 40 parameters: 20 angles and 20 radii (log-scaled to keep positive).
    """
    elapsed = time.time() - _fitness.start_time
    if elapsed > 580:
        return 0.0
    
    angles = params[:20]
    log_radii = params[20:]
    radii = np.exp(log_radii)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    
    return _compute_min_vertex_score(polygon)


def _regular_polygon():
    """Return the regular 20-gon on the unit circle."""
    n = 20
    angles = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that tries to be a counterexample to the
    conjecture "every convex polygon has a vertex v with no 4 other vertices equidistant
    from v". The goal is a polygon where EVERY vertex has (at least) 4 other vertices that
    are approximately equidistant from it. Quality is the minimum, over all vertices, of a
    per-vertex equidistance score in [0, 1] (1.0 = perfect counterexample).
    
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    _fitness.start_time = time.time()
    deadline = 580  # seconds
    
    # Start with regular polygon
    best_polygon = _regular_polygon()
    best_score = _compute_min_vertex_score(best_polygon)
    
    # Differential evolution over angles and log-radii
    n = 20
    # Angles in [0, 2*pi), log_radii in [-2, 2] (radii in [e^-2, e^2] ~ [0.135, 7.39])
    bounds = [(0, 2*np.pi)] * n + [(-2, 2)] * n
    
    # Try differential evolution
    try:
        result = differential_evolution(
            _fitness,
            bounds=bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-4,
            polish=False,
            init='latinhypercube',
            update_interval=1
        )
        
        if result.fun > best_score:
            params = result.x
            angles = params[:20]
            log_radii = params[20:]
            radii = np.exp(log_radii)
            x = radii * np.cos(angles)
            y = radii * np.sin(angles)
            candidate = np.stack([x, y], axis=1)
            score = _compute_min_vertex_score(candidate)
            if score > best_score:
                best_polygon = candidate
                best_score = score
    except Exception:
        pass
    
    elapsed = time.time() - _fitness.start_time
    if elapsed > deadline - 30:
        return best_polygon
    
    # Try multiple random restarts with local refinement
    for trial in range(5):
        elapsed = time.time() - _fitness.start_time
        if elapsed > deadline - 30:
            break
        
        # Random angles, radii near 1
        angles = np.sort(np.random.uniform(0, 2*np.pi, n))
        log_radii = np.random.uniform(-0.5, 0.5, n)
        radii = np.exp(log_radii)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        candidate = np.stack([x, y], axis=1)
        
        score = _compute_min_vertex_score(candidate)
        if score > best_score:
            best_polygon = candidate
            best_score = score
    
    return best_polygon


# EVOLVE-BLOCK-END
