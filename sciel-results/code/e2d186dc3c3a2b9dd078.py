# EVOLVE-BLOCK-START
import numpy as np
import time

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
    raw = np.mean(np.maximum(sel / D, D / sel))
    score = min(1.0, 1.0 / raw)
    return score

def _polygon_min_score(polygon):
    """
    Compute the minimum vertex score over all vertices of a polygon.
    polygon: (n, 2) array.
    """
    n = len(polygon)
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(centered ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    normalized = centered / mean_dist
    
    # Check convexity
    if np.any(np.linalg.norm(np.diff(np.vstack([normalized, normalized[:1]]), axis=0), axis=1) < 1e-10):
        return 0.0
    edges = np.diff(np.vstack([normalized, normalized[:1]]), axis=0)
    cross = edges[:, 0] * edges[:, 1] - edges[:, 1] * edges[:, 0]
    if np.any(np.abs(cross) < 1e-10):
        return 0.0
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return 0.0
    total_turn = np.sum(np.arctan2(cross, edges[:, 0] * edges[:, 0] + edges[:, 1] * edges[:, 1]))
    if abs(abs(total_turn) - 2 * np.pi) > 1e-6:
        return 0.0
    
    min_score = 1.0
    for i in range(n):
        rotated = np.roll(normalized, -i, axis=0)
        v_score = _compute_vertex_score(rotated)
        if v_score < min_score:
            min_score = v_score
    return min_score

def _make_radial_polygon(amp, freq, phase, n=20):
    """
    Create a polygon where radius r(θ) = 1 + amp * cos(freq * θ + phase).
    This keeps the polygon close to a circle, maintaining convexity for small amp.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = 1.0 + amp * np.cos(freq * angles + phase)
    # Ensure radii are positive
    radii = np.maximum(radii, 0.5)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _search_lowdim():
    """
    Search over low-dimensional parametric family:
    r(θ) = 1 + amp * cos(freq * θ + phase)
    Try several frequencies (2, 3, 4, 5) and fine grid over amp and phase.
    """
    n = 20
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin
    
    best_score = 0.0
    best_polygon = None
    
    # Regular polygon as baseline
    regular_poly = _make_radial_polygon(0.0, 2, 0.0, n)
    best_score = _polygon_min_score(regular_poly)
    best_polygon = regular_poly
    
    # Try different frequencies and amplitudes
    frequencies = [2, 3, 4, 5]
    
    for freq in frequencies:
        # Grid over amplitude
        for amp in np.linspace(0.0, 0.15, 16):
            if time.time() > deadline:
                break
            # Grid over phase
            for phase in np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False):
                if time.time() > deadline:
                    break
                poly = _make_radial_polygon(amp, freq, phase, n)
                score = _polygon_min_score(poly)
                if score > best_score:
                    best_score = score
                    best_polygon = poly.copy()
    
    # Local refinement around best found
    if best_polygon is not None and time.time() < deadline:
        # Extract best parameters by re-evaluating nearby
        best_amp, best_freq, best_phase = 0.0, 2, 0.0
        # We'll just do a finer local search around the best polygon found
        # by perturbing the polygon directly in a small neighborhood
        current_poly = best_polygon.copy()
        current_score = best_score
        
        for _ in range(200):
            if time.time() > deadline:
                break
            # Small random perturbation to radii
            angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
            # Get current radii
            r = np.sqrt(current_poly[:, 0]**2 + current_poly[:, 1]**2)
            theta = np.arctan2(current_poly[:, 1], current_poly[:, 0])
            # Perturb radii
            new_r = r + np.random.normal(0, 0.01, n)
            new_r = np.maximum(new_r, 0.5)
            new_poly = np.stack([new_r * np.cos(theta), new_r * np.sin(theta)], axis=1)
            new_score = _polygon_min_score(new_poly)
            if new_score > current_score:
                current_poly = new_poly
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_polygon = new_poly.copy()
    
    return best_polygon

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    Returns:
        polygon: np.ndarray of shape (20, 2) with the x,y coordinates of the vertices,
            given in convex (counter-clockwise) order.
    """
    start_time = time.time()
    
    polygon = _search_lowdim()
    
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
