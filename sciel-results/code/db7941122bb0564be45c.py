"""TSP constructive heuristic: select_next_node with destination-aware lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = None
_DEADLINE_MARGIN = 2.0  # seconds


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city in a TSP tour construction.
    
    Uses a destination-biased nearest neighbor with a small 2-step lookahead
    to balance local distance with progress toward the destination.
    
    Parameters:
        current_node: int - current city index
        destination_node: int - city to return to at the end
        unvisited_nodes: np.ndarray - array of unvisited city indices
        distance_matrix: np.ndarray - pairwise distance matrix
    
    Returns:
        int - index of the next city to visit
    """
    global _DEADLINE
    
    # Set deadline on first call
    if _DEADLINE is None:
        _DEADLINE = time.time() + 148.0  # safety margin
    
    # Check deadline
    if time.time() > _DEADLINE - _DEADLINE_MARGIN:
        # Fallback: just pick nearest neighbor
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(destination_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get distances from current to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from each unvisited to destination
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Simple score: weighted sum of distance from current and distance to destination
    # The weight on destination encourages progress toward the end of the tour
    # A weight of 0.5 means we care equally about how far we go now and how close
    # we get to the destination
    
    # Also add a small penalty based on the "regret" - how much worse the second-best
    # option is, which can help avoid getting stuck. But keep it simple for speed.
    
    # For small sets, use a 2-step lookahead:
    # For each candidate u, estimate the cost of going current -> u -> best_next(u)
    # where best_next(u) is the closest unvisited node to u (excluding u itself)
    # This gives a better approximation than just d(current,u) + w*d(u,dest)
    
    if n_unvisited <= 30:
        # 2-step lookahead
        # For each unvisited node u, find the min distance from u to any other unvisited node
        # This is O(n^2) per call, but n <= 30 so it's manageable
        unvisited_arr = unvisited_nodes
        # dist_matrix among unvisited: dists_from_current already has current->unvisited
        # We need unvisited->unvisited distances
        uu_dists = distance_matrix[np.ix_(unvisited_arr, unvisited_arr)]
        # Set diagonal to inf so we don't pick self
        np.fill_diagonal(uu_dists, np.inf)
        min_next_dist = np.min(uu_dists, axis=1)
        
        # Score: d(current, u) + 0.3 * d(u, dest) + 0.2 * min_dist(u, other_unvisited)
        # The min_dist term encourages picking nodes that have good connections to others
        scores = dists_from_current + 0.3 * dists_to_dest + 0.2 * min_next_dist
        
        min_idx = np.argmin(scores)
        return int(unvisited_nodes[min_idx])
    else:
        # For larger sets, use simpler heuristic for speed
        # Score: d(current, u) + 0.5 * d(u, dest)
        scores = dists_from_current + 0.5 * dists_to_dest
        min_idx = np.argmin(scores)
        return int(unvisited_nodes[min_idx])

# EVOLVE-BLOCK-END
