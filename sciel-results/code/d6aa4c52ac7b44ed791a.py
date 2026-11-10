# EVOLVE-BLOCK-START
import numpy as np
import time


def _vertex_score(v, all_pts):
    """Compute the score for a single vertex v given all polygon points."""
    # Distances to all other vertices
    diffs = all_pts - v
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    # Exclude self
    mask = np.ones(len(dists), dtype=bool)
    mask[np.argmin(dists)] = False  # self is closest (distance 0)
    d = dists[mask]
    d = np.sort(d)
    
    if len(d) < 4:
        return 0.0
    
    # Check minimum distance
    if d[0] < 0.05:
        return 0.0
    
    # Find k: first index minimizing d_{k+1} - d_k
    gaps = d[1:] - d[:-1]
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    
    # Select four distances
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k-1:k+3]
    
    # raw = mean of max(d/D, D/d)
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)


def _evaluate_polygon(pts):
    """
    Evaluate the polygon: normalize, check convexity, compute min vertex score.
    Returns the minimum vertex score (0 if invalid).
    """
    if len(pts) != 20:
        return 0.0
    
    # Check for repeated points
    if np.any(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2, axis=2) < 1e-10):
        return 0.0
    
    # Normalize: subtract centroid
    centroid = np.mean(pts, axis=0)
    pts_norm = pts - centroid
    
    # Mean distance to centroid
    dists_to_centroid = np.sqrt(np.sum(pts_norm ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    
    pts_norm = pts_norm / mean_dist
    
    # Check strict convexity: compute cross products of consecutive edges
    # For CCW polygon, all cross products should be positive (left turns)
    edges = np.roll(pts_norm, -1, axis=0) - pts_norm
    cross_products = np.cross(edges, np.roll(edges, -1, axis=0))
    
    # All cross products must be nonzero and of the same sign
    if np.any(np.abs(cross_products) < 1e-10):
        return 0.0
    
    # All must be same sign
    if not (np.all(cross_products > 0) or np.all(cross_products < 0)):
        return 0.0
    
    # Total turning should be ±2π (for a simple convex polygon, this is guaranteed
    # if all turns are same sign and nonzero, but let's verify)
    # Compute angles
    edge_lengths = np.linalg.norm(edges, axis=1)
    if np.any(edge_lengths < 1e-10):
        return 0.0
    
    # Compute turn angles
    cos_angles = np.sum(edges * np.roll(edges, -1, axis=0), axis=1) / (edge_lengths * np.roll(edge_lengths, -1, axis=1))
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    sin_angles = cross_products / (edge_lengths * np.roll(edge_lengths, -1, axis=1))
    turn_angles = np.arctan2(sin_angles, cos_angles)
    
    total_turn = np.sum(turn_angles)
    if abs(abs(total_turn) - 2 * np.pi) > 0.1:
        return 0.0
    
    # Compute vertex scores
    scores = [_vertex_score(pts_norm[i], pts_norm) for i in range(20)]
    return min(scores)


def _regular_polygon(radius=1.0):
    """Create a regular 20-gon."""
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def _perturb_polygon(pts, sigma=0.01):
    """Perturb a polygon by adding small noise to coordinates."""
    return pts + np.random.normal(0, sigma, size=pts.shape)


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices optimized for the equidistance score.
    
    Uses a combination of:
    1. Regular polygon as baseline
    2. Local search with coordinate perturbations
    3. Edge-length/angle parameterization for better exploration
    
    Returns:
        polygon: np.ndarray of shape (20, 2) with vertex coordinates in convex order.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget
    
    # Start with regular polygon
    best_pts = _regular_polygon(1.0)
    best_score = _evaluate_polygon(best_pts)
    
    # Local search with adaptive step size
    np.random.seed(42)
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 10:  # 10 second safety margin
        # Try a perturbation
        perturbed = _perturb_polygon(best_pts, step_size)
        score = _evaluate_polygon(perturbed)
        
        if score > best_score:
            best_pts = perturbed
            best_score = score
            no_improve_count = 0
            # Increase step size slightly when improving
            step_size *= 1.05
        else:
            no_improve_count += 1
            # Decrease step size when not improving
            step_size *= 0.95
            if no_improve_count > max_no_improve:
                # Reset to a fresh perturbation
                best_pts = _regular_polygon(1.0)
                best_score = _evaluate_polygon(best_pts)
                step_size = 0.05
                no_improve_count = 0
                max_no_improve = max(20, max_no_improve // 2)
    
    return best_pts


# EVOLVE-BLOCK-END
