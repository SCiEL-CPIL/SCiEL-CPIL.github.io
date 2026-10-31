"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for force-directed approach
_fd_start_time = None
_fd_deadline = 120.0  # seconds, safety margin before 150s kill
_fd_cache = {}


def _is_deadline_passed():
    """Check if we've exceeded our internal deadline."""
    if _fd_start_time is None:
        return False
    return (time.time() - _fd_start_time) > _fd_deadline


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in a TSP tour using a force-directed approach.
    
    Each unvisited city experiences:
    - Attractive force toward the current node (want to move forward)
    - Repulsive force from the destination (avoid overshooting)
    - A "cohesion" force that keeps the tour compact
    
    We select the candidate that best aligns with the direction from current to destination,
    while avoiding edge crossings by considering the geometric structure.
    """
    global _fd_start_time
    if _fd_start_time is None:
        _fd_start_time = time.time()
    
    if len(unvisited_nodes) == 0:
        return int(current_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline - if we're running out of time, fall back to nearest neighbor
    if _is_deadline_passed():
        dist_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dist_from_current)
        return int(unvisited_nodes[best_idx])
    
    # If we have only two unvisited nodes, use exact 2-opt style comparison
    if len(unvisited_nodes) == 2:
        n1, n2 = unvisited_nodes
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return int(n1 if cost1 <= cost2 else n2)
    
    # Force-directed scoring:
    # 1. Attractive component: distance from current to candidate (lower is better)
    # 2. Alignment component: how well the candidate lies in the "direction" toward destination
    #    This is computed using the angle between (current->candidate) and (current->destination)
    # 3. Cohesion component: prefer candidates that are close to other unvisited nodes
    #    to avoid creating long isolated edges
    
    # Extract coordinates implicitly from distance matrix using law of cosines
    # For alignment, we compute the "projection" of each candidate onto the current->dest direction
    
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    dist_current_dest = distance_matrix[current_node, destination_node]
    
    # Compute alignment score using the law of cosines:
    # For triangle (current, candidate, destination):
    # cos(angle) = (d_cc^2 + d_cd^2 - d_cand_dest^2) / (2 * d_cc * d_cd)
    # where d_cc = dist(current, candidate), d_cd = dist(current, destination)
    # A positive cos means the candidate is in the "forward" direction
    
    # Avoid division by zero
    safe_dist_from_current = np.where(dist_from_current > 1e-10, dist_from_current, 1e-10)
    safe_dist_current_dest = max(dist_current_dest, 1e-10)
    
    # cos(angle) = (d_cc^2 + d_cd^2 - d_cand_dest^2) / (2 * d_cc * d_cd)
    cos_angle = (dist_from_current**2 + dist_current_dest**2 - dist_to_dest**2) / \
                (2.0 * safe_dist_from_current * safe_dist_current_dest)
    # Clamp to [-1, 1] for numerical stability
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Alignment score: higher cos_angle means better alignment with destination direction
    # We want to maximize alignment, so we negate it for minimization
    alignment_score = -cos_angle
    
    # Cohesion: minimum distance from candidate to any other unvisited node
    # This prevents picking a node that would be far from the remaining cluster
    n_unvisited = len(unvisited_nodes)
    if n_unvisited > 2:
        sub_dist = distance_matrix[unvisited_nodes][:, unvisited_nodes]
        # Create a copy to avoid modifying the original
        sub_dist_work = sub_dist.copy()
        np.fill_diagonal(sub_dist_work, np.inf)
        min_cohesion = np.min(sub_dist_work, axis=1)
    else:
        min_cohesion = np.zeros(n_unvisited)
    
    # Combine forces:
    # Primary: distance from current (attractive force)
    # Secondary: alignment with destination direction (physics-based guidance)
    # Tertiary: cohesion to avoid isolated edges
    
    weight_attract = 1.0
    weight_align = 0.4
    weight_cohesion = 0.3
    
    # Normalize alignment to comparable scale
    # cos_angle is in [-1, 1], so alignment_score is in [-1, 1]
    # Scale it to be comparable to distances
    scale_factor = dist_current_dest if dist_current_dest > 1e-10 else 1.0
    alignment_score_scaled = alignment_score * scale_factor
    
    score = (weight_attract * dist_from_current + 
             weight_align * alignment_score_scaled + 
             weight_cohesion * min_cohesion)
    
    # Pick the candidate with the minimum combined score
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
