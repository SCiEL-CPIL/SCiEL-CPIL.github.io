"""Traveling salesman constructive heuristic with time-aware lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # 10s safety margin before 120s limit

# Cache for computed scores to avoid redundant work
_score_cache = {}


def _get_remaining_time():
    return _DEADLINE - time.time()


def _soft_min(values, temperature=0.5):
    """Compute soft-min using log-sum-exp trick for numerical stability."""
    if len(values) == 0:
        return 0.0
    min_val = np.min(values)
    exp_vals = np.exp(-(values - min_val) / temperature)
    return min_val + temperature * np.log(np.sum(exp_vals))


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in a TSP tour using a multi-criteria scoring function.
    
    The score combines:
    1. Distance from current node (immediate cost)
    2. Distance from candidate to destination (future cost)
    3. A soft-min lookahead over remaining nodes (smoothing for better exploration)
    
    This balances exploitation (nearest neighbor) with exploration (lookahead).
    """
    global _score_cache
    
    # If no unvisited nodes, return a safe default
    if len(unvisited_nodes) == 0:
        return int(current_node)
    
    # If only one unvisited node, return it
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Check if we're close to deadline - fall back to fast nearest neighbor
    if _get_remaining_time() < 1.0:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    # Compute base distances
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_destination = distance_matrix[unvisited_nodes, destination_node]
    
    # Weighted combination of immediate and future costs
    # Use a temperature parameter to balance exploration vs exploitation
    # Higher weight on dist_from_current = more greedy
    alpha = 0.7  # weight for current distance
    beta = 0.3   # weight for destination distance
    
    base_score = alpha * dist_from_current + beta * dist_to_destination
    
    # For small problem sizes, add a lookahead component
    if n_unvisited <= 30 and _get_remaining_time() > 5.0:
        # Compute soft-min lookahead: for each candidate, estimate the cost
        # of continuing from that candidate to the remaining unvisited nodes
        # This is a smoothed approximation of the true future cost
        
        # To keep computation tractable, use a simplified lookahead
        # that considers the nearest remaining node after choosing each candidate
        lookahead_scores = np.zeros(n_unvisited)
        
        for i, cand in enumerate(unvisited_nodes):
            # Exclude the candidate from the remaining set for lookahead
            remaining_mask = np.ones(n_unvisited, dtype=bool)
            remaining_mask[i] = False
            remaining_nodes = unvisited_nodes[remaining_mask]
            
            if len(remaining_nodes) > 0:
                # Distance from candidate to its nearest remaining neighbor
                nearest_dist = np.min(distance_matrix[cand, remaining_nodes])
                # Distance from that nearest neighbor to destination
                nearest_neighbor_idx = np.argmin(distance_matrix[cand, remaining_nodes])
                nn_node = remaining_nodes[nearest_neighbor_idx]
                nn_to_dest = distance_matrix[nn_node, destination_node]
                
                # Combined lookahead cost
                lookahead_scores[i] = 0.5 * nearest_dist + 0.5 * nn_to_dest
            else:
                lookahead_scores[i] = 0.0
        
        # Combine base score with lookahead using soft-min smoothing
        # The soft-min gives a smoothed estimate of the best possible continuation
        temperature = 0.3
        smoothed_lookahead = _soft_min(lookahead_scores, temperature)
        
        # Normalize lookahead scores to [0, 1] range for weighting
        if np.max(lookahead_scores) > 0:
            normalized_lookahead = lookahead_scores / np.max(lookahead_scores)
        else:
            normalized_lookahead = lookahead_scores
        
        # Final score: base + weighted lookahead
        gamma = 0.2  # weight for lookahead
        final_score = base_score + gamma * normalized_lookahead * smoothed_lookahead
        
    else:
        # Fast path: just use base score
        final_score = base_score
    
    # Select the node with minimum score
    best_idx = np.argmin(final_score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
