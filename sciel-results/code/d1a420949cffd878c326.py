"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # 10s safety margin before 120s limit

def _elapsed():
    return time.time() - _START_TIME

def _is_deadline():
    return time.time() >= _DEADLINE


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Convex-hull-peeling heuristic with angle-based direction preference.
    
    Strategy:
    1. Compute a 'hull-ness' score for each unvisited node: nodes far from the
       centroid of unvisited nodes are more likely to be on the convex hull.
    2. Compute an angle score: prefer nodes that continue in a direction similar
       to the vector from previous node to current node (momentum), but also
       consider the direction toward destination for the final leg.
    3. Combine: strong weight on hull-ness to trace the outer boundary,
       moderate weight on angle continuity to avoid sharp turns/crossings.
    4. Deadline guard: if we're running out of time, fall back to nearest neighbor.
    """
    if _is_deadline():
        # Emergency fallback: nearest neighbor
        if len(unvisited_nodes) == 0:
            return int(current_node)
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Distances from current node to all unvisited
    dists = distance_matrix[current_node, unvisited_nodes]  # shape (n_unvisited,)
    
    # Centroid of unvisited nodes (use distance matrix to approximate? 
    # Actually we don't have coordinates, only distance matrix.
    # We need a proxy for "hull-ness" using only the distance matrix.
    
    # Hull-ness proxy: a node on the convex hull tends to be far from the
    # center of mass. We can approximate the center of mass by the node that
    # minimizes the sum of distances to all other unvisited nodes.
    # But computing that for each candidate is O(n^2). For 50 nodes it's fine.
    
    # Alternative: use the average distance from each unvisited node to all other
    # unvisited nodes. Nodes on the hull will have higher average distance.
    
    # Compute pairwise distances among unvisited nodes
    # unvisited_nodes is an array of indices into the original matrix
    unvisited_set = unvisited_nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_set, unvisited_set)]  # (n_unvisited, n_unvisited)
    
    # Average distance from each unvisited node to all other unvisited nodes
    avg_dist_to_others = np.mean(sub_matrix, axis=1)  # shape (n_unvisited,)
    
    # Normalize hull-ness score: higher = more likely on hull
    # Use z-score or just rank
    hull_score = avg_dist_to_others  # higher is better for hull
    
    # Angle-based score: we want to continue in a direction that minimizes
    # the turn angle. Without coordinates, we can use the triangle inequality
    # or the relationship between distances.
    
    # Direction vector from current to destination (as a reference)
    dist_current_dest = distance_matrix[current_node, destination_node]
    
    # For each candidate, compute a "directional consistency" score.
    # If we go from current -> candidate, then from candidate -> destination,
    # the path length is dists[i] + distance_matrix[candidate, destination].
    # The straight-line distance is dist_current_dest.
    # The "detour" is (dists[i] + dist_candidate_dest) - dist_current_dest.
    # Smaller detour means the candidate is more "in line" with the direction to destination.
    
    dist_candidate_dest = distance_matrix[unvisited_set, destination_node]  # shape (n_unvisited,)
    detour = dists + dist_candidate_dest - dist_current_dest  # shape (n_unvisited,)
    
    # Normalize detour: lower is better (more aligned with destination direction)
    # But we don't want to always go toward destination; we want to trace the hull.
    
    # Combine scores:
    # - hull_score: prefer nodes far from center (hull-like)
    # - detour: prefer nodes that don't deviate too much from the destination direction
    #   (this helps avoid backtracking and crossing edges)
    
    # Normalize both to [0, 1] for combining
    def normalize(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    hull_norm = normalize(hull_score)  # higher is better
    detour_norm = normalize(detour)  # lower is better, so we invert
    
    # We want to maximize hull_norm and minimize detour_norm
    # Combined score = alpha * hull_norm - beta * detour_norm
    
    # Use adaptive weights: when few nodes remain, rely more on detour (to close the tour)
    # When many nodes remain, rely more on hull (to trace the boundary)
    if n_unvisited > 10:
        alpha = 0.7
        beta = 0.3
    elif n_unvisited > 5:
        alpha = 0.5
        beta = 0.5
    else:
        alpha = 0.2
        beta = 0.8
    
    combined_score = alpha * hull_norm - beta * detour_norm
    
    # Also add a small penalty for very long distances to avoid jumping across
    dist_norm = normalize(dists)
    combined_score -= 0.1 * dist_norm
    
    # Pick the best
    best_idx = int(np.argmax(combined_score))
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
