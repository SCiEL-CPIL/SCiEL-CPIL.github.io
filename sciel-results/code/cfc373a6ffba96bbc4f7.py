"""Traveling salesman constructive heuristic with geometric angle-based selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache for coordinates and timing
_coords_cache = None
_start_time = None


def _get_coordinates(distance_matrix):
    """Recover approximate 2D coordinates from the distance matrix using classical MDS."""
    global _coords_cache
    if _coords_cache is not None:
        return _coords_cache
    
    # Use classical MDS (Torgerson) to recover 2D coordinates
    # d^2 = ||x_i - x_j||^2
    D = distance_matrix
    D_sq = D ** 2
    
    # Center the matrix
    n = D_sq.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D_sq @ J
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top 2 eigenvalues
    eigenvalues = np.maximum(eigenvalues, 0)
    X = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    
    _coords_cache = X
    return X


def _init_timing():
    global _start_time
    if _start_time is None:
        _start_time = time.time()


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a geometric heuristic that balances:
    1. Angle deviation from the direction toward destination (prefers continuing in a consistent direction)
    2. Distance to current node (prefers closer nodes)
    3. Convex hull awareness (prefers boundary nodes to avoid edge crossings)
    """
    _init_timing()
    
    # Check time budget
    elapsed = time.time() - _start_time
    if elapsed > 140:
        # Safety: fall back to nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return destination_node
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    coords = _get_coordinates(distance_matrix)
    
    cur_pos = coords[current_node]
    dest_pos = coords[destination_node]
    cand_pos = coords[unvisited_nodes]
    
    # Vector from current to destination
    to_dest = dest_pos - cur_pos
    to_dest_norm = np.linalg.norm(to_dest)
    
    # Vectors from current to each candidate
    to_cands = cand_pos - cur_pos
    cand_dists = np.linalg.norm(to_cands, axis=1)
    
    # Handle zero-distance cases
    valid_mask = cand_dists > 1e-10
    if not np.all(valid_mask):
        # Fall back to nearest neighbor for degenerate cases
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Compute angles between to_dest and to_cands
    # cos(theta) = (to_dest . to_cand) / (|to_dest| * |to_cand|)
    dot_products = to_cands @ to_dest
    cos_angles = dot_products / (cand_dists * to_dest_norm + 1e-10)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.arccos(cos_angles)  # angle in [0, pi]
    
    # Angle score: prefer small angles (continuing in same direction)
    # But we also want to eventually come back, so use a balanced scoring
    angle_score = angles / (np.pi)  # normalize to [0, 1]
    
    # Distance score: prefer closer nodes, but not too close (to avoid getting stuck)
    max_dist = np.max(cand_dists)
    min_dist = np.min(cand_dists)
    dist_range = max_dist - min_dist + 1e-10
    dist_score = (cand_dists - min_dist) / dist_range  # normalize to [0, 1]
    
    # Convex hull awareness: prefer nodes that are far from the centroid of unvisited nodes
    # This encourages tracing the boundary
    centroid = np.mean(cand_pos, axis=0)
    to_centroid = cand_pos - centroid
    centroid_dists = np.linalg.norm(to_centroid, axis=1)
    max_cd = np.max(centroid_dists)
    min_cd = np.min(centroid_dists)
    cd_range = max_cd - min_cd + 1e-10
    hull_score = (centroid_dists - min_cd) / cd_range  # normalize to [0, 1]
    
    # Combined score: we want to minimize this
    # Weighting: angle is most important for reducing crossings, 
    # distance for efficiency, hull for boundary tracing
    w_angle = 0.5
    w_dist = 0.3
    w_hull = 0.2
    
    # For angle, we prefer small angles, so use angle_score directly
    # For distance, we prefer small distances, so use dist_score directly  
    # For hull, we prefer large hull_score (boundary nodes), so use (1 - hull_score)
    combined = w_angle * angle_score + w_dist * dist_score + w_hull * (1 - hull_score)
    
    # Also consider the distance from candidate to destination as a tiebreaker
    # to ensure we don't wander too far from the goal
    cand_to_dest = np.linalg.norm(cand_pos - dest_pos, axis=1)
    max_cd2 = np.max(cand_to_dest)
    min_cd2 = np.min(cand_to_dest)
    cd2_range = max_cd2 - min_cd2 + 1e-10
    dest_score = (cand_to_dest - min_cd2) / cd2_range
    
    combined += 0.1 * dest_score
    
    # Select the candidate with minimum combined score
    best_idx = np.argmin(combined)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
